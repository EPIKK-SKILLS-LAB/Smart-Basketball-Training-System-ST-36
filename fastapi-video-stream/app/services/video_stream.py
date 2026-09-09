from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import os

app = FastAPI()

VIDEO_DIRECTORY = "videos"

def stream_video(file_path: str):
    with open(file_path, "rb") as video_file:
        while chunk := video_file.read(1024):
            yield chunk

@app.get("/video/{video_name}")
async def get_video(video_name: str):
    file_path = os.path.join(VIDEO_DIRECTORY, video_name)
    if os.path.exists(file_path):
        return StreamingResponse(stream_video(file_path), media_type="video/mp4")
    return Response(status_code=404)