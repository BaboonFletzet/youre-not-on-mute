"""Convert official base visual checkpoint using a pinned Transformers port; strict weights."""
from pathlib import Path
import hashlib,json,sys,torch,sentencepiece as spm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/avhubert-s2s'))
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.model.av2text_config import AV2TextConfig
class Dictionary: pass
class BPE: pass
path=ROOT/'.cache/models/avhubert-base-vsr.pt'
# Only allow inert dictionary holders and SentencePiece's known processor class.
with torch.serialization.safe_globals([(Dictionary,'fairseq.data.dictionary.Dictionary'),
    (BPE,'fairseq.data.encoders.sentencepiece_bpe.SentencepieceBPE'),spm.SentencePieceProcessor]):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
weights=checkpoint['model'];dictionary=checkpoint['task_state']['target_dictionary']
config=json.loads((ROOT/'.cache/avhubert-hf-config.txt').read_text())
config.update(d_model=768,encoder_hidden_size=768,encoder_embed_dim=768,decoder_hidden_size=768,
    decoder_layers=6,decoder_ffn_dim=3072,decoder_attention_heads=4,encoder_attention_heads=12,
    intermediate_size=3072,num_hidden_layers=12,encoder_layers=12,num_classes=weights['encoder.w2v_model.label_embs_concat'].shape[0],
    vocab_size=len(dictionary.symbols),bos_token_id=dictionary.bos_index,pad_token_id=dictionary.pad_index,
    eos_token_id=dictionary.eos_index,decoder_start_token_id=dictionary.eos_index,scale_embedding=False,
    max_target_positions=1024,decoder_attn_implementation='eager',attn_implementation='eager')
model=AV2TextForConditionalGeneration(AV2TextConfig(**config))
replacements={'pos_conv.0':'pos_conv_embed.conv','self_attn.k_proj':'attention.k_proj',
'self_attn.v_proj':'attention.v_proj','self_attn.q_proj':'attention.q_proj','self_attn.out_proj':'attention.out_proj',
'self_attn_layer_norm':'layer_norm','fc1':'feed_forward.intermediate_dense','fc2':'feed_forward.output_dense'}
converted={};ignored=[]
for key,value in weights.items():
    if key.startswith('encoder.w2v_model.'):
        key='model.encoder.'+key.removeprefix('encoder.w2v_model.')
        for old,new in replacements.items():key=key.replace(old,new)
    elif key.startswith('decoder.'):
        key='model.'+key
    else:raise ValueError(f'Unmapped key {key}')
    if key.endswith(('._float_tensor','.version')):
        ignored.append(key);continue
    if key=='model.decoder.output_projection.weight':
        converted['lm_head.weight']=value
        continue
    converted[key]=value
# HF's model wrapper ties these weights to the same decoder embedding.
converted['model.lm_head.weight']=converted['model.decoder.embed_tokens.weight']
converted['lm_head.weight']=converted['model.decoder.embed_tokens.weight']
# Sinusoidal positions are deterministic buffers, absent from Fairseq checkpoints.
converted['model.decoder.embed_positions.weights']=model.model.decoder.embed_positions.weights.detach().clone()
missing,unexpected=model.load_state_dict(converted,strict=False)
print('missing',missing,'unexpected',unexpected,flush=True)
if missing or unexpected:raise RuntimeError('Conversion incomplete; refusing to save a partially initialized model')
dst=ROOT/'.cache/models/avhubert-base-hf';dst.mkdir(exist_ok=True)
# Untied copies are explicit, with no pickle objects required at runtime.
from safetensors.torch import save_file
save_file({k:v.detach().float().contiguous().clone() for k,v in model.state_dict().items()},str(dst/'model.safetensors'))
model.config.save_pretrained(dst)
(dst/'tokens.json').write_text(json.dumps(dictionary.symbols))
(dst/'conversion.json').write_text(json.dumps({'source':'official base_vox_433h.pt',
    'source_sha256':'84c83698c831b447cb75db0d14b72040c021c9a442ee1208d306b1091c00756d','ignored_nonparameter_keys':ignored},indent=2))
print('saved',dst,flush=True)
