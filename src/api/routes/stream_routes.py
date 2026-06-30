"""
Stream routes: snapshot endpoint for live grid camera view.

Returns the latest frame from the camera frame buffer as JPEG.
Falls back to a placeholder 204 response when no frame is available
(e.g., during development without an active ingestion pipeline).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()


@router.get("/cameras/{camera_id}/snapshot", tags=["stream"])
async def get_camera_snapshot(camera_id: str) -> Response:
    """Return the latest JPEG frame for a camera, or 204 if unavailable.

    In production this should pull from the ingestion frame buffer.
    For now returns 204 so the frontend img tag gracefully shows the
    placeholder error state.
    """
    # TODO (Phase 13+): Integrate with IngestionManager frame buffer.
    # frame = ingestion_manager.get_latest_frame(camera_id)
    # if frame is not None:
    #     import cv2
    #     _, jpeg = cv2.imencode(".jpg", frame)
    #     return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    return Response(status_code=204)
