from TTS.api import TTS

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Check if the model supports multiple speakers
supports_multiple_speakers = hasattr(tts, "speakers")

if supports_multiple_speakers:
    available_speakers = tts.speakers
    print(f"Available speakers: {available_speakers}")
else:
    print("The model does not support multiple speakers.")
