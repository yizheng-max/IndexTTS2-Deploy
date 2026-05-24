import os
import sys

# Set HF cache to local checkpoints dir (contains pre-cached models)
os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_download')
os.environ['HF_HUB_CACHE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'hub')
os.environ['TORCH_HOME'] = r'D:\AI_TTS_Workspace\cache\torch'
os.environ['XDG_CACHE_HOME'] = r'D:\AI_TTS_Workspace\cache'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indextts.infer_v2 import IndexTTS2

# Test inference
prompt_wav = "examples/voice_01.wav"
text = "大家好，欢迎体验IndexTTS2.0语音合成系统。"
output_path = "outputs/output_test.wav"

os.makedirs("outputs", exist_ok=True)

print(">> Initializing IndexTTS2...")
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    is_fp16=False,
    device="cpu"
)

print(f">> Running inference...")
tts.infer(
    prompt_wav,
    text,
    output_path,
    verbose=True
)
print(f">> Output saved to: {output_path}")
