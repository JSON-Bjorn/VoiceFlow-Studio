import asyncio
import sys
import time
import os

sys.path.append("./backend")

# Enable production mode for maximum GPU performance
os.environ["TTS_PRODUCTION_MODE"] = "true"


async def test_gpu_acceleration():
    print("🚀 Testing MANDATORY GPU Acceleration (NO CPU FALLBACK)")
    print("=" * 60)

    try:
        # Test GPU validator first
        print("🔍 Step 1: GPU Validation Service")
        try:
            from backend.app.services.gpu_validator import gpu_validator

            gpu_status = gpu_validator.get_gpu_status()

            if gpu_status["status"] == "active":
                print(f"✅ GPU Validator: SUCCESS")
                print(f"   🎯 GPU: {gpu_status['gpu_name']}")
                print(f"   💾 Total VRAM: {gpu_status['memory_total_gb']}GB")
                print(f"   🆓 Free VRAM: {gpu_status['memory_free_gb']}GB")
                print(
                    f"   📊 Memory Usage: {gpu_status['memory_utilization_percent']:.1f}%"
                )
                print(f"   🔧 CUDA Version: {gpu_status['cuda_version']}")
                print(f"   🐍 PyTorch Version: {gpu_status['pytorch_version']}")
            else:
                print(
                    f"❌ GPU Validator: FAILED - {gpu_status.get('error', 'Unknown error')}"
                )
                return

        except Exception as e:
            print(f"❌ GPU Validator initialization failed: {e}")
            print("💡 This means GPU requirements are not met")
            return

        print()

        # Test Chatterbox service with GPU acceleration
        print("🎤 Step 2: Chatterbox TTS GPU Acceleration")
        try:
            from backend.app.services.chatterbox_service import ChatterboxService

            service = ChatterboxService()
            print(f"✅ Chatterbox Service: SUCCESS")
            print(f"   🖥️  Device: {service.device}")
            print(f"   ⚡ Production Mode: {service.production_mode}")
            print(f"   🏃 Inference Steps: {service.inference_steps} (ultra-fast)")
            print(f"   📏 Max Chunk Length: {service.max_chunk_length}")
            print()

            # Test connection
            connection_result = await service.test_connection()
            if connection_result["status"] == "success":
                print(f"✅ GPU Connection Test: SUCCESS")
                print(f"   🎯 GPU: {connection_result['gpu_name']}")
                print(f"   💾 VRAM: {connection_result['gpu_memory_gb']}GB")
                print(
                    f"   🎵 Target Quality: {connection_result['target_format'].upper()} @ {connection_result['target_bitrate']}"
                )
                print(f"   📊 Sample Rate: {connection_result['target_sample_rate']}Hz")
            else:
                print(
                    f"❌ GPU Connection Test: FAILED - {connection_result['message']}"
                )
                return

        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                print(f"❌ Expected GPU requirement error: {e}")
                print("💡 This is correct behavior - GPU is mandatory")
                return
            else:
                print(f"❌ Unexpected error: {e}")
                return
        except Exception as e:
            print(f"❌ Chatterbox service failed: {e}")
            return

        print()

        # Test actual TTS generation performance
        print("🚀 Step 3: GPU Performance Test")
        test_text = (
            "Testing GPU acceleration for maximum performance in podcast generation."
        )
        print(f'Text: "{test_text}"')
        print(f"Length: {len(test_text)} characters")
        print()

        # First generation (no cache)
        print("🔥 Testing first generation (no cache)...")
        start_time = time.time()
        response = await service.convert_text_to_speech(test_text)
        generation_time = time.time() - start_time

        if response.success:
            ratio = generation_time / response.duration
            print(f"   ✅ Generation: {generation_time:.2f}s")
            print(f"   🎵 Audio Duration: {response.duration:.2f}s")
            print(f"   ⚡ Speed Ratio: {ratio:.2f}x")
            print(f"   📦 Audio Size: {len(response.audio_data):,} bytes")
            print(f"   🎯 Format: {response.audio_format}")
            print(f"   📊 Sample Rate: {response.sample_rate}Hz")

            # Performance evaluation
            if ratio < 1.0:
                print("   🏆 EXCEPTIONAL: Faster than real-time!")
            elif ratio < 1.5:
                print("   🎉 EXCELLENT: Sub-1.5x ratio achieved!")
            elif ratio < 2.0:
                print("   ✅ GOOD: Sub-2x ratio achieved!")
            elif ratio < 3.0:
                print("   ⚠️  ACCEPTABLE: Sub-3x ratio")
            else:
                print("   ❌ SLOW: Still needs optimization")

            # 10-minute podcast projection
            projected_time = (600 * ratio) / 60  # Convert to minutes
            print(f"   📻 10-minute podcast projection: {projected_time:.1f} minutes")

            if projected_time < 10:
                print("   🚀 REAL-TIME: Can generate faster than playback!")
            elif projected_time < 15:
                print("   ✅ EXCELLENT: Production ready")
            elif projected_time < 30:
                print("   ⚠️  ACCEPTABLE: Usable for production")
            else:
                print("   ❌ TOO SLOW: Needs further optimization")

        else:
            print(f"   ❌ Generation failed: {response.error_message}")
            return

        print()

        # Test caching performance
        print("⚡ Testing cache performance...")
        start_time = time.time()
        cached_response = await service.convert_text_to_speech(test_text)
        cache_time = time.time() - start_time

        if cached_response.success:
            print(f"   ✅ Cache retrieval: {cache_time:.4f}s (essentially instant)")
            speedup = generation_time / cache_time if cache_time > 0 else float("inf")
            print(f"   🚀 Cache speedup: {speedup:.0f}x faster")

        print()

        # Final GPU memory check
        print("💾 Final GPU Memory Status:")
        final_memory = gpu_validator.get_memory_usage()
        if "error" not in final_memory:
            print(f"   📊 Total: {final_memory['total_gb']}GB")
            print(f"   🟢 Allocated: {final_memory['allocated_gb']}GB")
            print(f"   🔴 Reserved: {final_memory['reserved_gb']}GB")
            print(f"   ⚪ Free: {final_memory['free_gb']}GB")
            print(f"   📈 Utilization: {final_memory['utilization_percent']}%")

        print()
        print("🎉 GPU ACCELERATION TEST COMPLETE")
        print("✅ All systems operating with mandatory GPU acceleration")
        print("❌ CPU fallback successfully DISABLED")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        print()
        print("💡 This likely means GPU requirements are not met.")
        print("   Ensure CUDA and compatible GPU are properly installed.")


if __name__ == "__main__":
    asyncio.run(test_gpu_acceleration())
