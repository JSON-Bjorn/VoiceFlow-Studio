from typing import Dict, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates during podcast generation
    """

    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[int, List[WebSocket]] = {}
        # Store generation sessions by generation_id
        self.generation_sessions: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept new WebSocket connection"""
        await websocket.accept()

        if user_id not in self.active_connections:
            self.active_connections[user_id] = []

        self.active_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")

    async def send_progress_update(
        self,
        user_id: int,
        generation_id: str,
        phase: str,
        progress: int,
        message: str,
        metadata: Optional[Dict] = None,
    ):
        """Send progress update to all user's connections"""
        if user_id not in self.active_connections:
            return

        update_data = {
            "type": "progress_update",
            "generation_id": generation_id,
            "phase": phase,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        # Update session data
        if generation_id not in self.generation_sessions:
            self.generation_sessions[generation_id] = {
                "user_id": user_id,
                "started_at": datetime.utcnow().isoformat(),
                "current_phase": phase,
                "progress": progress,
                "history": [],
            }

        session = self.generation_sessions[generation_id]
        session["current_phase"] = phase
        session["progress"] = progress
        session["last_update"] = datetime.utcnow().isoformat()
        session["history"].append(
            {
                "phase": phase,
                "progress": progress,
                "message": message,
                "timestamp": update_data["timestamp"],
            }
        )

        # Send to all user connections
        disconnected = []
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_text(json.dumps(update_data))
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.append(websocket)

        # Remove disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, user_id)

    async def send_generation_complete(
        self, user_id: int, generation_id: str, success: bool, result: Dict[str, Any]
    ):
        """Send generation completion notification"""
        if user_id not in self.active_connections:
            return

        completion_data = {
            "type": "generation_complete",
            "generation_id": generation_id,
            "success": success,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Update session
        if generation_id in self.generation_sessions:
            session = self.generation_sessions[generation_id]
            session["completed_at"] = datetime.utcnow().isoformat()
            session["success"] = success
            session["final_result"] = result

        # Send to all user connections
        disconnected = []
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_text(json.dumps(completion_data))
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error sending completion message: {e}")
                disconnected.append(websocket)

        # Remove disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, user_id)

    async def send_error_notification(
        self,
        user_id: int,
        generation_id: str,
        error_message: str,
        error_details: Optional[Dict] = None,
    ):
        """Send error notification"""
        if user_id not in self.active_connections:
            return

        error_data = {
            "type": "generation_error",
            "generation_id": generation_id,
            "error_message": error_message,
            "error_details": error_details or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Send to all user connections
        disconnected = []
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_text(json.dumps(error_data))
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error sending error message: {e}")
                disconnected.append(websocket)

        # Remove disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, user_id)

    def get_generation_status(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a generation session"""
        return self.generation_sessions.get(generation_id)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old generation sessions"""
        current_time = datetime.utcnow()
        to_remove = []

        for generation_id, session in self.generation_sessions.items():
            session_time = datetime.fromisoformat(session["started_at"])
            age_hours = (current_time - session_time).total_seconds() / 3600

            if age_hours > max_age_hours:
                to_remove.append(generation_id)

        for generation_id in to_remove:
            del self.generation_sessions[generation_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old generation sessions")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
