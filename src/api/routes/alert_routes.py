"""Alert routes."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.auth import get_current_user_payload
from src.api.dependencies import get_alert_repo
from src.api.repositories import AlertRepository
from src.api.schemas.alert_schemas import (
    AlertAcknowledge, AlertDismiss, AlertListResponse, AlertResponse,
)

router = APIRouter()


@router.get("/stats")
def alert_stats(
    repo: AlertRepository = Depends(get_alert_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Return aggregated alert statistics."""
    return repo.get_alert_stats()


@router.get("/active", response_model=List[AlertResponse])
def active_alerts(
    repo: AlertRepository = Depends(get_alert_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Return only pending/delivered (unacknowledged) alerts."""
    records, _ = repo.list_alerts(limit=200)
    return [r for r in records if r.status in ("pending", "delivered")]


@router.get("", response_model=AlertListResponse)
def list_alerts(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    repo: AlertRepository = Depends(get_alert_repo),
    _: dict = Depends(get_current_user_payload),
):
    """List alerts with optional filtering."""
    records, total = repo.list_alerts(
        status=status, priority=priority, limit=limit, offset=offset,
    )
    return AlertListResponse(alerts=records, total=total, limit=limit, offset=offset)


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: str,
    repo: AlertRepository = Depends(get_alert_repo),
    _: dict = Depends(get_current_user_payload),
):
    """Get a specific alert."""
    record = repo.get_alert(alert_id)
    if not record:
        raise HTTPException(status_code=404, detail="Alert not found")
    return record


@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    body: AlertAcknowledge,
    repo: AlertRepository = Depends(get_alert_repo),
    payload: dict = Depends(get_current_user_payload),
):
    """Acknowledge an alert."""
    record = repo.update_status(
        alert_id, "acknowledged", operator=body.operator or payload.get("sub")
    )
    if not record:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"acknowledged": alert_id}


@router.post("/{alert_id}/dismiss")
def dismiss_alert(
    alert_id: str,
    body: AlertDismiss,
    repo: AlertRepository = Depends(get_alert_repo),
    payload: dict = Depends(get_current_user_payload),
):
    """Dismiss an alert."""
    record = repo.update_status(
        alert_id, "dismissed", operator=body.operator or payload.get("sub")
    )
    if not record:
        raise HTTPException(status_code=404, detail="Alert not found")
    record.dismiss_reason = body.reason
    return {"dismissed": alert_id}
