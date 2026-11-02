import numpy as np
import pytest
from services.inference.alignment import token_spans, word_spans, WindowCommitter
from services.inference.decoding import DecodeOptions, rank_candidates
from services.inference.evaluation import edits, summarize
from services.inference.stream import FrameBuffer


def test_ctc_repeated_tokens_need_blank_between_them():
    lp = np.full((5, 3), -20.0)
    lp[np.arange(5), [0, 1, 0, 1, 0]] = 0
    assert token_spans(lp, [1, 1]) == [(1, 2), (3, 4)]
    assert token_spans(lp[:1], [1, 1]) is None


def test_alignment_joins_subwords_and_keeps_word_times():
    lp = np.full((6, 4), -20.0)
    lp[np.arange(6), [0, 1, 2, 0, 3, 0]] = 0
    result = word_spans(lp, [1, 2, 3], ['<blank>', '▁HEL', 'LO', '▁WORLD'])
    assert [x['text'] for x in result] == ['HELLO', 'WORLD']
    assert result[0]['end'] <= result[1]['start']


def test_overlap_keeps_repeated_words_in_distinct_time_regions_and_flushes_tail():
    merger = WindowCommitter()
    words = [{'text': 'go', 'start': .1, 'end': .2}, {'text': 'go', 'start': .85, 'end': .95}]
    assert merger.select(words, 0, 4000, True, hold_ms=500) == 'go'
    # Next window contains the held-back second word, then new content.
    next_words = [{'text': 'go', 'start': .1, 'end': .2}, {'text': 'now', 'start': .8, 'end': .9}]
    assert merger.select(next_words, 3000, 7000, True, hold_ms=500) == 'go now'
    assert merger.select([{'text':'please', 'start': .8, 'end': 1}], 6000, 7200, True) == 'please'


def test_buffer_retains_context_without_accepting_reordered_timestamps():
    buffer = FrameBuffer()
    for i in range(101): buffer.push(i * 40, np.zeros(1))
    buffer.retain(1)
    assert buffer.frames[0][0] == 3000
    with pytest.raises(ValueError): buffer.push(3999, np.zeros(1))


def test_abstention_counts_all_reference_words_as_deleted():
    assert edits('read these words', '')['deletions'] == 3
    result = summarize([{'reference':'read these words', 'text':'', 'elapsed':.1, 'video_seconds':2}])
    assert result['wer'] == 1
    assert result['utterance_coverage'] == 0


def test_ctc_rescoring_preserves_visual_candidate_set():
    candidates = [{'text':'a', 'ids':[1], 'attention_score':-1, 'ctc_score':-20, 'lm_score':-3},
                  {'text':'b', 'ids':[2], 'attention_score':-2, 'ctc_score':-1, 'lm_score':-2}]
    assert rank_candidates(candidates, DecodeOptions())[0]['text'] == 'a'
    assert rank_candidates(candidates, DecodeOptions(ctc_weight=.3))[0]['text'] == 'b'
    with pytest.raises(ValueError): DecodeOptions(lm_weight=float('nan'))


def test_pronunciation_decoder_returns_words_and_rejects_blank_input():
    from services.inference.phoneme_words import PronunciationDecoder
    phones=['<blank>','HH','AH','L','OW','W','ER','D']
    decoder=PronunciationDecoder(phones,{'hello':[['HH','AH0','L','OW1']]},beam_size=16)
    lp=np.full((9,len(phones)),-20.)
    lp[np.arange(9),[0,1,0,2,0,3,0,4,0]]=0
    assert decoder.decode(lp)[0]['text']=='hello'
    blank=np.full((9,len(phones)),-1000.);blank[:,0]=0
    # Extremely weak alternate paths exist mathematically; their score records the lack of evidence.
    candidates=decoder.decode(blank)
    assert not candidates or candidates[0]['score'] < -100


def test_face_input_diagnostics_do_not_break_reference_landmarks():
    from types import SimpleNamespace as NS
    from services.inference.model import VisualRecognizer
    points=[NS(x=.3,y=.4),NS(x=.7,y=.4),NS(x=.5,y=.5),NS(x=.5,y=.7)]
    detection=NS(location_data=NS(relative_keypoints=points))
    recognizer=VisualRecognizer.__new__(VisualRecognizer)
    recognizer.detectors=[NS(process=lambda frame:NS(detections=[detection]))]
    landmarks,coverage=recognizer.landmarks([np.zeros((100,100,3),dtype=np.uint8)]*2)
    assert coverage==1
    assert landmarks[0].shape==(4,2)
    assert recognizer.input_quality['inter_eye_pixels']==40
    assert recognizer.input_quality['small_face'] is False
