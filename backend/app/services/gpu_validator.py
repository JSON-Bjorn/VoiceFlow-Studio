"""
GPU Validation Service

This service ensures mandatory GPU acceleration across the entire application.
CPU-based processing is disabled for production performance requirements.
"""

import logging
import torch
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class GPUValidator:
    """
    GPU validation service to enforce mandatory GPU acceleration.

    This service ensures that:
    1. CUDA/GPU is available and properly configured
    2. Sufficient GPU memory is available
    3. All audio processing uses GPU acceleration
    4. No CPU fallback is allowed for production performance
    """

    def __init__(self):
        """Initialize the GPU validator"""
        self.minimum_vram_gb = 4  # Minimum 4GB VRAM required
        self.validate_gpu_requirements()

    def validate_gpu_requirements(self) -> None:
        """Validate that GPU requirements are met for the application"""
        logger.info("🔍 Validating GPU requirements for production performance...")

        # Check CUDA availability
        if not torch.cuda.is_available():
            error_msg = (
                "❌ CRITICAL: CUDA/GPU not available! "
                "This application requires GPU acceleration for production performance. "
                "CPU-based TTS processing is 4-5x slower (40+ minutes for 10-minute podcasts). "
                "\n\nRequired setup:"
                "\n1. Install NVIDIA GPU drivers"
                "\n2. Install CUDA toolkit (compatible with PyTorch)"
                "\n3. Install PyTorch with CUDA support"
                "\n4. Ensure sufficient GPU memory (minimum 4GB VRAM)"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get GPU information
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu_properties.total_memory // 1024**3

        # Validate GPU memory
        if gpu_memory_gb < self.minimum_vram_gb:
            error_msg = (
                f"❌ Insufficient GPU memory! "
                f"Found: {gpu_memory_gb}GB VRAM, Required: {self.minimum_vram_gb}GB minimum. "
                f"Audio generation requires significant GPU memory for optimal performance."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Log successful validation
        logger.info(f"✅ GPU validation successful!")
        logger.info(f"   🎯 GPU: {gpu_name}")
        logger.info(
            f"   💾 VRAM: {gpu_memory_gb}GB (minimum {self.minimum_vram_gb}GB required)"
        )
        logger.info(f"   🔢 GPU Count: {gpu_count}")
        logger.info(f"   📱 CUDA Version: {torch.version.cuda}")
        logger.info(f"   🐍 PyTorch Version: {torch.__version__}")
        logger.info("🚀 Mandatory GPU acceleration enabled - CPU fallback DISABLED")

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and performance metrics"""
        try:
            if not torch.cuda.is_available():
                return {
                    "status": "error",
                    "available": False,
                    "error": "CUDA not available",
                    "recommendation": "Install CUDA and compatible GPU drivers",
                }

            gpu_properties = torch.cuda.get_device_properties(0)
            memory_total = gpu_properties.total_memory
            memory_reserved = torch.cuda.memory_reserved(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_free = memory_total - memory_reserved

            return {
                "status": "active",
                "available": True,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_total_gb": memory_total // 1024**3,
                "memory_allocated_gb": memory_allocated // 1024**3,
                "memory_reserved_gb": memory_reserved // 1024**3,
                "memory_free_gb": memory_free // 1024**3,
                "memory_utilization_percent": (memory_allocated / memory_total) * 100,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "gpu_compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                "multiprocessors": gpu_properties.multi_processor_count,
            }

        except Exception as e:
            return {
                "status": "error",
                "available": False,
                "error": str(e),
                "recommendation": "Check GPU drivers and CUDA installation",
            }

    def ensure_gpu_available(self) -> None:
        """Ensure GPU is still available during runtime"""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ GPU acceleration lost during runtime! Cannot continue without CUDA."
            )

    def optimize_gpu_settings(self) -> None:
        """Apply optimal GPU settings for audio generation"""
        try:
            if torch.cuda.is_available():
                # Enable cuDNN optimizations
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = (
                    True  # Optimize for consistent input sizes
                )

                # Clear GPU cache to free memory
                torch.cuda.empty_cache()

                logger.info("⚡ Applied GPU optimizations:")
                logger.info("   ✅ cuDNN enabled and optimized")
                logger.info("   ✅ GPU memory cache cleared")

        except Exception as e:
            logger.warning(f"⚠️  Could not apply some GPU optimizations: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        memory_free = memory_total - memory_reserved

        return {
            "total_gb": round(memory_total, 2),
            "allocated_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2),
            "free_gb": round(memory_free, 2),
            "utilization_percent": round((memory_allocated / memory_total) * 100, 1),
        }

    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache to free up VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("🧹 GPU memory cache cleared")


# Global GPU validator instance
gpu_validator = GPUValidator()
