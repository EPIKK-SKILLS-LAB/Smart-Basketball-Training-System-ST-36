from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.model_service import process_frame

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
VIDEO_DIRECTORY = BASE_DIR / "videos"
RECORDINGS_DIRECTORY = BASE_DIR / "recordings"

_connections: set[WebSocket] = set()

_dashboard_state: dict[str, Any] = {
    "type": "feedback",
    "result": "WAITING",
    "shot": 0,
    "acc": 0,
    "made": 0,
    "missed": 0,
    "airball": 0,
    "time": "00:00:00",
    "state": "IDLE",
    "conn": "STANDALONE",
    "ir": {
        "1": 0,
        "2": 0,
        "3": 0,
    },
}


class FeedbackPayload(BaseModel):
    type: str = "feedback"
    result: str = "WAITING"
    shot: int = Field(default=0, ge=0)
    acc: float = Field(default=0, ge=0, le=100)
    made: int = Field(default=0, ge=0)
    missed: int = Field(default=0, ge=0)
    airball: int = Field(default=0, ge=0)
    time: str = "00:00:00"
    state: str = "IDLE"
    conn: str = "CONNECTED"
    ir: dict[str, int] = Field(
        default_factory=lambda: {
            "1": 0,
            "2": 0,
            "3": 0,
        }
    )


def ensure_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"{path} exists as a file. Rename or delete it.")

    path.mkdir(parents=True, exist_ok=True)


ensure_directory(VIDEO_DIRECTORY)
ensure_directory(RECORDINGS_DIRECTORY)


def dashboard_payload() -> dict[str, Any]:
    return {
        **_dashboard_state,
        "ir": {
            "1": _dashboard_state["ir"].get("1", 0),
            "2": _dashboard_state["ir"].get("2", 0),
            "3": _dashboard_state["ir"].get("3", 0),
        },
    }


async def broadcast_feedback(payload: dict[str, Any]) -> None:
    disconnected = []

    for websocket in _connections:
        try:
            await websocket.send_json(payload)
        except Exception:
            disconnected.append(websocket)

    for websocket in disconnected:
        _connections.discard(websocket)


@router.post("/feedback")
async def receive_feedback(feedback: FeedbackPayload):
    """
    Receives the collective session values from the model/device and
    broadcasts them to every connected dashboard.
    """
    global _dashboard_state

    values = feedback.model_dump()
    values["type"] = "feedback"

    _dashboard_state = values

    await broadcast_feedback(dashboard_payload())

    return {
        "ok": True,
        "feedback": dashboard_payload(),
    }


@router.get("/feedback")
def get_feedback():
    """Returns the latest collective session values."""
    return dashboard_payload()


@router.websocket("/training/ws")
async def training_websocket(websocket: WebSocket):
    await websocket.accept()
    _connections.add(websocket)

    recording_id = uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    recording_path = (
        RECORDINGS_DIRECTORY
        / f"training_{timestamp}_{recording_id}.avi"
    )

    writer = None
    frame_count = 0

    try:
        start_message = await websocket.receive_json()

        if start_message.get("type") != "start":
            await websocket.close(code=1003)
            return

        await websocket.send_json({
            "type": "started",
            "recording_id": recording_id,
            **dashboard_payload(),
        })

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") == "stop":
                break

            frame_bytes = message.get("bytes")

            if not frame_bytes:
                continue

            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(recording_path),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    10.0,
                    (width, height),
                )

            writer.write(frame)
            frame_count += 1

            model_feedback = process_frame(frame)

            if frame_count % 3 == 0:
                payload = dashboard_payload()

                # Preserve the collective dashboard values while exposing
                # the current frame-level model feedback.
                payload["model"] = model_feedback
                payload["frame"] = frame_count

                await websocket.send_json(payload)

        await websocket.send_json({
            "type": "completed",
            "file": recording_path.name if writer else None,
            "frames": frame_count,
            **dashboard_payload(),
        })

    except WebSocketDisconnect:
        pass

    finally:
        _connections.discard(websocket)

        if writer is not None:
            writer.release()


@router.get("/records")
def list_records():
    records = []

    for file in sorted(
        RECORDINGS_DIRECTORY.glob("*.avi"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        records.append({
            "name": file.name,
            "url": f"/api/records/{file.name}",
            "size": file.stat().st_size,
        })

    return {"records": records}


@router.get("/records/{record_name}")
def download_record(record_name: str):
    record_path = (RECORDINGS_DIRECTORY / record_name).resolve()

    if RECORDINGS_DIRECTORY not in record_path.parents:
        raise HTTPException(status_code=400, detail="Invalid record name")

    if not record_path.is_file():
        raise HTTPException(status_code=404, detail="Record not found")

    return FileResponse(
        record_path,
        media_type="video/x-msvideo",
        filename=record_path.name,
    )


@router.get("/video/{video_name}")
async def stream_video(video_name: str):
    video_path = (VIDEO_DIRECTORY / video_name).resolve()

    if VIDEO_DIRECTORY not in video_path.parents:
        raise HTTPException(status_code=400, detail="Invalid video name")

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(video_path, media_type="video/mp4")