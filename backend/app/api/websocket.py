from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
)
from ..services.websocket_manager import websocket_manager
from ..core.auth import get_current_user_from_token, get_current_user
from ..models.user import User
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/progress/{token}")
async def websocket_progress_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time progress updates during podcast generation

    URL: ws://localhost:8000/ws/progress/{jwt_token}

    Messages sent to client:
    - progress_update: Real-time generation progress
    - generation_complete: Generation finished successfully
    - generation_error: Generation failed with error
    """
    try:
        # Verify JWT token and get user
        user = await get_current_user_from_token(token)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Connect to WebSocket manager
        await websocket_manager.connect(websocket, user.id)
        logger.info(f"WebSocket connected for user {user.id} ({user.email})")

        try:
            # Keep connection alive and handle incoming messages
            while True:
                # Wait for any client messages (ping/pong, status requests, etc.)
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "ping":
                        # Respond to ping with pong
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "pong",
                                    "timestamp": websocket_manager.generation_sessions,
                                }
                            )
                        )

                    elif message_type == "get_status":
                        # Send current generation status if requested
                        generation_id = message.get("generation_id")
                        if generation_id:
                            status_data = websocket_manager.get_generation_status(
                                generation_id
                            )
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "status_response",
                                        "generation_id": generation_id,
                                        "status": status_data,
                                        "timestamp": websocket_manager.generation_sessions,
                                    }
                                )
                            )

                except json.JSONDecodeError:
                    # Invalid JSON, send error
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "Invalid JSON format"})
                    )

        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket, user.id)
            logger.info(f"WebSocket disconnected for user {user.id}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


@router.get("/ws/status/{generation_id}")
async def get_generation_status(
    generation_id: str, current_user: User = Depends(get_current_user)
):
    """
    HTTP endpoint to get current generation status (fallback for non-WebSocket clients)
    """
    status_data = websocket_manager.get_generation_status(generation_id)

    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Generation session not found"
        )

    # Verify user owns this generation
    if status_data.get("user_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    return {"success": True, "generation_id": generation_id, "status": status_data}
