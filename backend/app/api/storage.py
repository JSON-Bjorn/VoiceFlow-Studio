"""
Storage API Endpoints

Handles file serving and storage management operations.
"""

from fastapi import APIRouter, HTTPException, Depends, status, Response
from fastapi.responses import FileResponse
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..core.auth import get_current_user
from ..models.user import User
from ..services.storage_service import storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serve files from local storage

    This endpoint serves audio files and other stored files.
    In production, this should be handled by a CDN or web server like nginx.
    """
    try:
        # Security check - prevent directory traversal
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path"
            )

        # Get file data
        file_data = await storage_service.get_audio_file(file_path)

        # Determine content type based on file extension
        extension = Path(file_path).suffix.lower()
        content_type_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".json": "application/json",
            ".txt": "text/plain",
        }
        content_type = content_type_map.get(extension, "application/octet-stream")

        # Return file with appropriate headers
        return Response(
            content=file_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={Path(file_path).name}",
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            },
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error serving file",
        )


@router.get("/podcast/{podcast_id}/files")
async def list_podcast_files(
    podcast_id: str, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all files for a specific podcast

    Returns file paths and metadata for all files associated with a podcast.
    """
    try:
        files = await storage_service.list_podcast_files(podcast_id)

        # Get file URLs for each file
        file_info = []
        for file_path in files:
            try:
                url = await storage_service.get_file_url(file_path)
                file_info.append(
                    {"path": file_path, "url": url, "filename": Path(file_path).name}
                )
            except Exception as e:
                logger.warning(f"Could not get URL for file {file_path}: {str(e)}")

        return {
            "podcast_id": podcast_id,
            "files": file_info,
            "total_files": len(file_info),
        }

    except Exception as e:
        logger.error(f"Error listing files for podcast {podcast_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving file list",
        )


@router.delete("/files/{file_path:path}")
async def delete_file(
    file_path: str, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a file from storage

    Removes both the file and its metadata.
    """
    try:
        # Security check
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file path"
            )

        success = await storage_service.delete_audio_file(file_path)

        if success:
            return {
                "success": True,
                "message": f"File {file_path} deleted successfully",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete file",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting file",
        )


@router.get("/stats")
async def get_storage_stats(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get storage statistics

    Returns information about storage usage, file counts, etc.
    """
    try:
        stats = storage_service.get_storage_stats()
        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error getting storage stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving storage statistics",
        )


@router.post("/cleanup/temp")
async def cleanup_temp_files(
    older_than_hours: int = 24, current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Clean up temporary files

    Removes temporary files older than specified hours.
    """
    try:
        if older_than_hours < 1 or older_than_hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="older_than_hours must be between 1 and 168",
            )

        cleaned_count = await storage_service.cleanup_temp_files(older_than_hours)

        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} temporary files",
            "files_cleaned": cleaned_count,
            "older_than_hours": older_than_hours,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error cleaning up temporary files",
        )
