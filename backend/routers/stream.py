import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from backend.core.security import decode_access_token
from backend.services.detection_service import DetectionService

router = APIRouter()

# WebSocket guardrails
MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10 MB per frame
MAX_FRAMES_PER_SECOND = 30
MAX_CONCURRENT_STREAMS = 8
_connected_streams: dict[str, int] = {}


@router.websocket("/stream")
async def stream(ws: WebSocket, token: str = Query(...)):
    try:
        payload = decode_access_token(token)
    except Exception:
        await ws.close(code=4001, reason="Invalid token")
        return

    user_id = int(payload["sub"])
    conn_id = str(uuid.uuid4())

    # Enforce max concurrent streams
    if len(_connected_streams) >= MAX_CONCURRENT_STREAMS:
        await ws.close(code=4003, reason="Too many concurrent streams")
        return

    await ws.accept()
    _connected_streams[conn_id] = user_id
    frame_idx = 0
    last_frame_time = 0.0

    try:
        while True:
            frame_bytes = await ws.receive_bytes()

            # Enforce max frame size
            if len(frame_bytes) > MAX_FRAME_SIZE:
                await ws.send_json({"error": "Frame too large"})
                continue

            # Enforce frame rate throttle
            now = time.monotonic()
            min_interval = 1.0 / MAX_FRAMES_PER_SECOND
            if now - last_frame_time < min_interval:
                continue
            last_frame_time = now

            start = time.monotonic()
            result = await DetectionService.process_stream_frame(frame_bytes)
            latency = int((time.monotonic() - start) * 1000)

            await ws.send_json({
                "frame_index": frame_idx,
                "pedestrian_count": result["pedestrian_count"],
                "detections": result["detections"],
                "inference_ms": latency,
            })
            frame_idx += 1

    except WebSocketDisconnect:
        pass
    except Exception:
        await ws.close(code=1011, reason="Internal error")
    finally:
        _connected_streams.pop(conn_id, None)