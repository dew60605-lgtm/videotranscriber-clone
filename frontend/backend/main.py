from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import yt_dlp
import subprocess
import tempfile
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(youtube_url: str = Form(None), video: UploadFile = File(None)):
    with tempfile.TemporaryDirectory() as tmp:
        audio = os.path.join(tmp, "a.wav")
        if youtube_url:
            ydl_opts = {'format': 'bestaudio/best', 'outtmpl': os.path.join(tmp, 'a.%(ext)s'),
                        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}], 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([youtube_url])
            for f in os.listdir(tmp):
                if f.endswith('.wav'): audio = os.path.join(tmp, f); break
        elif video:
            v = os.path.join(tmp, "v.mp4")
            with open(v, "wb") as f: f.write(await video.read())
            subprocess.run(["ffmpeg", "-i", v, "-ar", "16000", "-ac", "1", "-f", "wav", audio], check=True)
        else:
            return {"detail": "No input provided"}
        segs, _ = model.transcribe(audio, beam_size=5)
        return {"text": "".join(s.text for s in segs).strip()}
