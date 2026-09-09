from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.routes.video import router as video_router

app = FastAPI(title="Smart Basketball Training System")

app.include_router(video_router, prefix="/api")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def home():
    return TEMPLATE_PATH.read_text(encoding="utf-8")