"""
Storage Service

Handles file storage operations for VoiceFlow Studio.
Currently implements local storage with easy migration path to AWS S3.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime
import hashlib
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    async def save_file(
        self, file_data: bytes, file_path: str, metadata: Dict[str, Any] = None
    ) -> str:
        """Save file and return the storage path/URL"""
        pass

    @abstractmethod
    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file data"""
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete file and return success status"""
        pass

    @abstractmethod
    async def get_file_url(self, file_path: str) -> str:
        """Get publicly accessible URL for file"""
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix filter"""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend"""

    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.audio_path = self.base_path / "audio"
        self.temp_path = self.base_path / "temp"
        self.metadata_path = self.base_path / "metadata"

        for path in [self.audio_path, self.temp_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)

    async def save_file(
        self, file_data: bytes, file_path: str, metadata: Dict[str, Any] = None
    ) -> str:
        """Save file to local storage"""
        try:
            full_path = self.base_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(full_path, "wb") as f:
                f.write(file_data)

            # Save metadata if provided
            if metadata:
                metadata_file = self.metadata_path / f"{file_path}.json"
                metadata_file.parent.mkdir(parents=True, exist_ok=True)

                # Add system metadata
                metadata.update(
                    {
                        "created_at": datetime.utcnow().isoformat(),
                        "file_size": len(file_data),
                        "file_hash": hashlib.sha256(file_data).hexdigest(),
                        "local_path": str(full_path),
                    }
                )

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            logger.info(f"Saved file to local storage: {file_path}")
            return str(full_path)

        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {str(e)}")
            raise

    async def get_file(self, file_path: str) -> bytes:
        """Retrieve file from local storage"""
        try:
            full_path = self.base_path / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(full_path, "rb") as f:
                return f.read()

        except Exception as e:
            logger.error(f"Failed to retrieve file {file_path}: {str(e)}")
            raise

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local storage"""
        try:
            full_path = self.base_path / file_path
            metadata_file = self.metadata_path / f"{file_path}.json"

            # Delete main file
            if full_path.exists():
                full_path.unlink()

            # Delete metadata file
            if metadata_file.exists():
                metadata_file.unlink()

            logger.info(f"Deleted file from local storage: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False

    async def get_file_url(self, file_path: str) -> str:
        """Get URL for local file (will be served by FastAPI)"""
        return f"/api/storage/files/{file_path}"

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix"""
        try:
            search_path = self.base_path / prefix if prefix else self.base_path
            files = []

            if search_path.exists():
                for file_path in search_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith("."):
                        # Return relative path from base
                        rel_path = file_path.relative_to(self.base_path)
                        files.append(str(rel_path))

            return sorted(files)

        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {str(e)}")
            return []


class StorageService:
    """Main storage service that handles file operations"""

    def __init__(self, backend: StorageBackend = None):
        self.backend = backend or LocalStorageBackend()

    async def save_audio_file(
        self,
        audio_data: bytes,
        podcast_id: str,
        segment_id: str = None,
        file_type: str = "mp3",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Save audio file with organized path structure"""

        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_id = str(uuid.uuid4())[:8]

        if segment_id:
            filename = f"{segment_id}_{timestamp}_{file_id}.{file_type}"
            file_path = f"audio/{podcast_id}/segments/{filename}"
        else:
            filename = f"podcast_{timestamp}_{file_id}.{file_type}"
            file_path = f"audio/{podcast_id}/{filename}"

        # Prepare metadata
        file_metadata = {
            "podcast_id": podcast_id,
            "segment_id": segment_id,
            "file_type": file_type,
            "original_filename": filename,
            **(metadata or {}),
        }

        storage_path = await self.backend.save_file(
            audio_data, file_path, file_metadata
        )
        return file_path

    async def save_temp_file(self, file_data: bytes, extension: str = "tmp") -> str:
        """Save temporary file"""
        file_id = str(uuid.uuid4())
        file_path = f"temp/{file_id}.{extension}"

        await self.backend.save_file(file_data, file_path)
        return file_path

    async def get_audio_file(self, file_path: str) -> bytes:
        """Retrieve audio file"""
        return await self.backend.get_file(file_path)

    async def delete_audio_file(self, file_path: str) -> bool:
        """Delete audio file"""
        return await self.backend.delete_file(file_path)

    async def get_file_url(self, file_path: str) -> str:
        """Get publicly accessible URL for file"""
        return await self.backend.get_file_url(file_path)

    async def list_podcast_files(self, podcast_id: str) -> List[str]:
        """List all files for a specific podcast"""
        return await self.backend.list_files(f"audio/{podcast_id}")

    async def cleanup_temp_files(self, older_than_hours: int = 24) -> int:
        """Clean up temporary files older than specified hours"""
        temp_files = await self.backend.list_files("temp")
        cleaned = 0

        cutoff_time = datetime.utcnow().timestamp() - (older_than_hours * 3600)

        for file_path in temp_files:
            try:
                full_path = Path(self.backend.base_path) / file_path
                if full_path.stat().st_mtime < cutoff_time:
                    if await self.backend.delete_file(file_path):
                        cleaned += 1
            except Exception as e:
                logger.error(f"Failed to check/delete temp file {file_path}: {str(e)}")

        logger.info(f"Cleaned up {cleaned} temporary files")
        return cleaned

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            if isinstance(self.backend, LocalStorageBackend):
                base_path = self.backend.base_path

                def get_dir_size(path: Path) -> int:
                    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

                stats = {
                    "backend_type": "local",
                    "base_path": str(base_path),
                    "total_size_bytes": get_dir_size(base_path),
                    "audio_size_bytes": get_dir_size(base_path / "audio"),
                    "temp_size_bytes": get_dir_size(base_path / "temp"),
                    "audio_files_count": len(list((base_path / "audio").rglob("*.*"))),
                    "temp_files_count": len(list((base_path / "temp").rglob("*.*"))),
                }

                # Convert bytes to human readable
                for key in ["total_size_bytes", "audio_size_bytes", "temp_size_bytes"]:
                    bytes_val = stats[key]
                    if bytes_val < 1024:
                        stats[key.replace("_bytes", "_human")] = f"{bytes_val} B"
                    elif bytes_val < 1024**2:
                        stats[key.replace("_bytes", "_human")] = (
                            f"{bytes_val / 1024:.1f} KB"
                        )
                    elif bytes_val < 1024**3:
                        stats[key.replace("_bytes", "_human")] = (
                            f"{bytes_val / (1024**2):.1f} MB"
                        )
                    else:
                        stats[key.replace("_bytes", "_human")] = (
                            f"{bytes_val / (1024**3):.1f} GB"
                        )

                return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {"error": str(e)}


# Global storage service instance
storage_service = StorageService()


# Helper functions for easy import
async def save_audio_file(audio_data: bytes, podcast_id: str, **kwargs) -> str:
    """Helper function to save audio file"""
    return await storage_service.save_audio_file(audio_data, podcast_id, **kwargs)


async def get_audio_file(file_path: str) -> bytes:
    """Helper function to get audio file"""
    return await storage_service.get_audio_file(file_path)


async def get_file_url(file_path: str) -> str:
    """Helper function to get file URL"""
    return await storage_service.get_file_url(file_path)
