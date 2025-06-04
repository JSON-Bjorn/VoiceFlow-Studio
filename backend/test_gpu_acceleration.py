#!/usr/bin/env python3
"""
GPU Acceleration Test for Chatterbox TTS

This script demonstrates the dramatic performance improvement when using
GPU acceleration compared to CPU-only processing for Chatterbox TTS.
"""

import asyncio
import sys
import os
import time
from pathlib import Path
import traceback

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

try:
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS

    CHATTERBOX_AVAILABLE = True
    print("✅ Chatterbox TTS libraries imported successfully")
except ImportError as e:
    CHATTERBOX_AVAILABLE = False
    print(f"❌ Chatterbox TTS not available: {e}")


async def check_gpu_availability():
    """Check GPU availability and system specs"""
    print("\n🔧 GPU Availability Check")
    print("=" * 50)

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory}GB")
        return True
    else:
        print("No CUDA GPU detected")
        return False


async def benchmark_generation_speed(device: str, text: str):
    """Benchmark text generation speed on specific device"""
    print(f"\n🚀 Benchmarking on {device.upper()}")
    print("=" * 50)

    try:
        # Load model
        print(f"Loading Chatterbox TTS on {device}...")
        load_start = time.time()
        model = ChatterboxTTS.from_pretrained(device=device)
        load_time = time.time() - load_start

        print(f"✅ Model loaded in {load_time:.2f} seconds")

        # Generate audio
        print(f"Generating audio ({len(text)} characters)...")
        gen_start = time.time()
        wav = model.generate(text)
        gen_time = time.time() - gen_start

        # Calculate metrics
        duration = wav.shape[1] / model.sr
        real_time_factor = duration / gen_time
        chars_per_second = len(text) / gen_time

        # Save output
        output_dir = Path("storage/audio/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gpu_test_{device}.wav"
        ta.save(str(output_file), wav, model.sr)

        results = {
            "device": device,
            "load_time": load_time,
            "generation_time": gen_time,
            "audio_duration": duration,
            "real_time_factor": real_time_factor,
            "chars_per_second": chars_per_second,
            "output_file": output_file,
        }

        print(f"✅ Generation completed!")
        print(f"   Audio duration: {duration:.2f} seconds")
        print(f"   Generation time: {gen_time:.2f} seconds")
        print(f"   Real-time factor: {real_time_factor:.1f}x")
        print(f"   Speed: {chars_per_second:.1f} chars/sec")
        print(f"   Saved: {output_file}")

        return results

    except Exception as e:
        print(f"❌ Benchmarking failed on {device}: {e}")
        return None


async def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("🎯 CHATTERBOX TTS - GPU ACCELERATION PERFORMANCE TEST")
    print("=" * 70)

    if not CHATTERBOX_AVAILABLE:
        print("❌ Chatterbox TTS not available")
        return

    # Check GPU availability
    gpu_available = await check_gpu_availability()

    # Test text - substantial enough to show performance difference
    test_text = """
    The evolution of artificial intelligence has reached a remarkable milestone with the advent of 
    local voice synthesis technologies like Chatterbox TTS. This groundbreaking advancement represents 
    more than just technological progress—it embodies a fundamental shift toward computational autonomy 
    and creative independence.
    
    When we consider the implications of GPU-accelerated voice generation, we're witnessing the 
    democratization of sophisticated audio production capabilities. No longer constrained by external 
    API limitations or recurring costs, creators and developers can explore the full spectrum of 
    voice synthesis applications without financial barriers or privacy concerns.
    
    The performance gains achieved through CUDA acceleration are truly transformational. What once 
    required minutes of processing time can now be accomplished in mere seconds, enabling real-time 
    applications and interactive experiences that were previously impossible. This acceleration 
    unlocks new possibilities for conversational AI, interactive media, and immersive storytelling.
    """.strip()

    print(f"\nTest text length: {len(test_text)} characters")

    results = []

    # Test on GPU (if available)
    if gpu_available:
        gpu_result = await benchmark_generation_speed("cuda", test_text)
        if gpu_result:
            results.append(gpu_result)

    # Test on CPU for comparison
    cpu_result = await benchmark_generation_speed("cpu", test_text)
    if cpu_result:
        results.append(cpu_result)

    # Performance comparison
    if len(results) >= 2:
        print("\n📊 PERFORMANCE COMPARISON")
        print("=" * 70)

        gpu_result = next(r for r in results if r["device"] == "cuda")
        cpu_result = next(r for r in results if r["device"] == "cpu")

        speedup_factor = cpu_result["generation_time"] / gpu_result["generation_time"]
        efficiency_improvement = (
            gpu_result["real_time_factor"] / cpu_result["real_time_factor"]
        )

        print(f"GPU Generation Time:  {gpu_result['generation_time']:.2f} seconds")
        print(f"CPU Generation Time:  {cpu_result['generation_time']:.2f} seconds")
        print(f"🚀 GPU SPEEDUP: {speedup_factor:.1f}x FASTER!")

        print(f"\nGPU Real-time Factor: {gpu_result['real_time_factor']:.1f}x")
        print(f"CPU Real-time Factor: {cpu_result['real_time_factor']:.1f}x")
        print(f"🎯 EFFICIENCY GAIN: {efficiency_improvement:.1f}x better")

        print(f"\nGPU Speed: {gpu_result['chars_per_second']:.1f} chars/sec")
        print(f"CPU Speed: {cpu_result['chars_per_second']:.1f} chars/sec")

        print(f"\n💡 PRACTICAL BENEFITS:")
        print(f"   ⚡ {speedup_factor:.1f}x faster podcast generation")
        print(
            f"   🎙️ Real-time synthesis capability: {'YES' if gpu_result['real_time_factor'] >= 1.0 else 'NO'}"
        )
        print(f"   💰 Zero additional API costs")
        print(f"   🔒 Complete privacy (local processing)")
        print(f"   🎨 Unlimited voice cloning experiments")

    elif len(results) == 1:
        result = results[0]
        device = result["device"].upper()
        print(f"\n📊 {device} PERFORMANCE RESULTS")
        print("=" * 50)
        print(f"Generation time: {result['generation_time']:.2f} seconds")
        print(f"Real-time factor: {result['real_time_factor']:.1f}x")
        print(f"Processing speed: {result['chars_per_second']:.1f} chars/sec")

        if device == "GPU":
            print(f"\n🎉 GPU acceleration is working!")
            print(
                f"   Real-time synthesis: {'YES' if result['real_time_factor'] >= 1.0 else 'NO'}"
            )

    else:
        print("❌ No successful benchmarks completed")
        return

    print(f"\n🎧 Generated audio files available in:")
    print(f"   storage/audio/test_outputs/")

    print(f"\n✨ CONCLUSION:")
    if gpu_available and len(results) >= 2:
        print(
            f"   GPU acceleration provides {speedup_factor:.0f}x performance improvement!"
        )
        print(f"   Your RTX 4060 dramatically enhances Chatterbox TTS capabilities.")
    else:
        print(f"   Chatterbox TTS is working, consider GPU for dramatic speedup!")


if __name__ == "__main__":
    print("Starting GPU acceleration performance test...\n")
    asyncio.run(run_performance_comparison())
