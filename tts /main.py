from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import subprocess
import os

app = FastAPI()

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

class SpeechRequest(BaseModel):
    text: str
    language: str = "en"
    speed: float = 1.1
    speaker: str = "0"  # Default speaker

@app.post("/generate_and_adjust_speech/")
async def generate_and_adjust_speech(request: SpeechRequest):
    try:
        # Generate speech from text
        input_file = "elon_clone.wav"
        tts.tts_to_file(
            text=request.text,
            file_path=input_file,
            language=request.language,
            speed=request.speed,
            speaker=request.speaker  # Pass the speaker here
        )

        # Adjust speed using FFmpeg through subprocess
        output_file = "elon_clone_fast.wav"
        subprocess.run([
            "ffmpeg", "-i", input_file, "-filter:a", f"atempo={request.speed}", output_file
        ], check=True)

        # Optionally, remove the original file if not needed
        os.remove(input_file)

        return {"message": f"Speech generated and speed adjusted. File saved as {output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4456)
