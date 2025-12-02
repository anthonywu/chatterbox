import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.utils.torch_load_patcher import torch_load_with_device

# Use the context manager instead of manual patching
with torch_load_with_device():  # Auto-detects MPS on Mac
    model = ChatterboxTTS.from_pretrained()
    
    text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."
    
    # If you want to synthesize with a different voice, specify the audio prompt
    AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
    wav = model.generate(
        text, 
        audio_prompt_path=AUDIO_PROMPT_PATH,
        exaggeration=2.0,
        cfg_weight=0.5
    )
    ta.save("test-2.wav", wav, model.sr)