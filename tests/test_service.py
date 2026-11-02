import struct
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from services import server


class FakeRecognizer:
    def set_decoder_profile(self, profile):
        self.decoder_profile = profile
    def recognize(self, frames):
        return {'text': 'test output', 'quality': 'readable', 'coverage': 1, 'elapsed': 0.01, 'device': 'test'}
    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    async def fake_load():
        server.recognizer = FakeRecognizer()
        server.state.update(status='ready', device='test')
    # The process-wide executor is not shut down between tests.
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(server, 'executor', ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(server, 'load_model', fake_load)
    monkeypatch.setattr(server, 'TOKEN', 'test-secret')
    monkeypatch.setattr(server, 'active', False)
    with TestClient(server.app) as c:
        yield c


def packet(timestamp, epoch=0):
    _, jpeg = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
    return struct.pack('<Id', epoch, timestamp) + jpeg.tobytes()


def test_rejects_untrusted_origin(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/stream', headers={'origin': 'https://untrusted.example'}):
            pass


def test_rejects_invalid_local_credential(client):
    with client.websocket_connect('/stream', headers={'origin': 'null'}) as ws:
        ws.send_json({'token': 'incorrect'})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008


def test_rejects_unsupported_language(client):
    with client.websocket_connect('/stream', headers={'origin': 'null'}) as ws:
        ws.send_json({'token': 'test-secret', 'language': 'fr'})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1008


def test_flush_decodes_short_clip_and_reset_ignores_old_frames(client):
    with client.websocket_connect('/stream', headers={'origin': 'null'}) as ws:
        ws.send_json({'token': 'test-secret', 'language': 'en'})
        assert ws.receive_json()['type'] == 'ready'
        for i in range(20):
            ws.send_bytes(packet(i * 40))
        ws.send_json({'type': 'reset', 'epoch': 1})
        assert ws.receive_json() == {'type': 'reset', 'epoch': 1}
        # Stale timestamps deliberately overlap the new epoch: they must be ignored.
        ws.send_bytes(packet(10000, epoch=0))
        for i in range(20):
            ws.send_bytes(packet(i * 40, epoch=1))
        ws.send_json({'type': 'flush'})
        result = ws.receive_json()
        assert result['type'] == 'caption'
        assert result['epoch'] == 1
        assert result['start_ms'] == 0
        assert result['end_ms'] == 760
        assert result['final'] is True


def test_frame_dimension_limit_returns_error(client):
    with client.websocket_connect('/stream', headers={'origin': 'null'}) as ws:
        ws.send_json({'token': 'test-secret'})
        ws.receive_json()
        _, jpeg = cv2.imencode('.jpg', np.zeros((1000, 1000, 3), dtype=np.uint8))
        ws.send_bytes(struct.pack('<Id', 0, 0) + jpeg.tobytes())
        assert ws.receive_json()['type'] == 'error'


def test_health_exposes_no_credentials(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['visual_only'] is True
    assert 'test-secret' not in response.text


def test_finish_drains_last_short_interval_before_acknowledgement(client):
    with client.websocket_connect('/stream', headers={'origin':'null'}) as ws:
        ws.send_json({'token':'test-secret'})
        assert ws.receive_json()['type']=='ready'
        for index in range(20): ws.send_bytes(packet(index*40))
        ws.send_json({'type':'finish'})
        result=ws.receive_json()
        assert result['type']=='caption'
        assert result['final'] is True
        assert result['end_ms']==760
        assert ws.receive_json()=={'type':'finished','epoch':0}


def test_finish_without_frames_acknowledges_without_inventing_caption(client):
    with client.websocket_connect('/stream', headers={'origin':'null'}) as ws:
        ws.send_json({'token':'test-secret'})
        ws.receive_json()
        ws.send_json({'type':'finish'})
        assert ws.receive_json()=={'type':'finished','epoch':0}
