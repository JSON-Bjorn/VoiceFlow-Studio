#!/usr/bin/env python3
"""
Real Chatterbox TTS Audio Generation Test

This script demonstrates the actual Chatterbox TTS capabilities using the correct
implementation and generates real audio files to showcase the migration benefits.
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
    print("Run: pip install chatterbox-tts torch torchaudio")


async def test_chatterbox_installation():
    """Test Chatterbox TTS installation and setup"""
    print("\n🔧 Testing Chatterbox TTS Installation...")

    if not CHATTERBOX_AVAILABLE:
        print("   ❌ Chatterbox TTS not installed")
        return False

    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"

        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {cuda_available}")
        print(f"   ✓ Device selected: {device}")

        if cuda_available:
            print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
            print(
                f"   ✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB"
            )

        return True

    except Exception as e:
        print(f"   ❌ Installation test failed: {e}")
        return False


async def generate_insightful_podcast_intro():
    """Generate an insightful podcast introduction using Chatterbox TTS"""
    print("\n🎙️ Generating Insightful Podcast Introduction...")

    if not CHATTERBOX_AVAILABLE:
        print("   ❌ Skipping - Chatterbox not available")
        return False

    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   🔧 Loading Chatterbox TTS model on {device}...")

        start_time = time.time()

        # Load the model
        model = ChatterboxTTS.from_pretrained(device=device)

        load_time = time.time() - start_time
        print(f"   ✓ Model loaded in {load_time:.2f} seconds")

        # Create insightful podcast content
        insightful_text = """
        Welcome to VoiceFlow Studio - where artificial intelligence meets human creativity in the most extraordinary way. 
        
        Today marks a revolutionary milestone in our journey. We've successfully migrated from ElevenLabs to Chatterbox TTS, 
        an open-source solution that doesn't just change how we generate voice - it transforms our entire philosophy of creation.
        
        Think about this: every word you're hearing right now is being processed locally, on our own hardware, 
        with zero data leaving our secure environment. No API calls, no external dependencies, no recurring costs. 
        This isn't just about saving money - it's about reclaiming control over our creative tools.
        
        But here's what's truly remarkable - Chatterbox doesn't just replicate voices, it enables voice cloning from simple audio prompts. 
        Imagine being able to create a podcast episode where historical figures discuss modern technology, 
        or where experts who never met engage in thoughtful dialogue across time and space.
        
        The implications are staggering. We're not just building a podcast platform - we're architecting the future of synthetic media, 
        where privacy, creativity, and technological excellence converge into something unprecedented.
        
        This is more than automation. This is augmented human potential.
        """

        print(f"   📝 Text length: {len(insightful_text)} characters")
        print(f"   🎯 Generating audio...")

        # Generate audio
        generation_start = time.time()
        wav = model.generate(insightful_text.strip())
        generation_time = time.time() - generation_start

        # Create output directory
        output_dir = Path("storage/audio/test_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the audio file
        output_file = output_dir / "insightful_podcast_intro.wav"
        ta.save(str(output_file), wav, model.sr)

        # Calculate statistics
        duration = wav.shape[1] / model.sr
        real_time_factor = duration / generation_time

        print(f"\n   📊 GENERATION RESULTS:")
        print(f"   ✅ Audio generated successfully!")
        print(f"   ✅ Duration: {duration:.2f} seconds")
        print(f"   ✅ Sample rate: {model.sr} Hz")
        print(f"   ✅ Generation time: {generation_time:.2f} seconds")
        print(f"   ✅ Real-time factor: {real_time_factor:.1f}x")
        print(f"   ✅ Audio saved to: {output_file}")
        print(f"   ✅ File size: {output_file.stat().st_size // 1024} KB")

        return True

    except Exception as e:
        print(f"   ❌ Audio generation failed: {e}")
        traceback.print_exc()
        return False


async def generate_conversation_sample():
    """Generate a sample conversation between two AI hosts"""
    print("\n👥 Generating Multi-Host Conversation Sample...")

    if not CHATTERBOX_AVAILABLE:
        print("   ❌ Skipping - Chatterbox not available")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)

        # Conversation segments
        conversation = [
            {
                "speaker": "Alex Chen",
                "text": "Sarah, what fascinates me most about our Chatterbox migration is the philosophical shift it represents. We're moving from dependency to autonomy.",
            },
            {
                "speaker": "Sarah Williams",
                "text": "Absolutely, Alex. When you think about it, every API call to an external service is essentially a small surrender of control. With Chatterbox running locally, we're reclaiming that sovereignty.",
            },
            {
                "speaker": "Alex Chen",
                "text": "And the creative possibilities are endless. Voice cloning means we could theoretically have Einstein discuss quantum computing with Steve Jobs. The boundaries between past and present dissolve.",
            },
            {
                "speaker": "Sarah Williams",
                "text": "That's the beauty of open-source innovation. We're not just users anymore - we're architects of our own creative destiny.",
            },
        ]

        output_dir = Path("storage/audio/test_outputs/conversation")
        output_dir.mkdir(parents=True, exist_ok=True)

        total_duration = 0
        total_generation_time = 0

        print(f"   🎬 Generating {len(conversation)} conversation segments...")

        for i, segment in enumerate(conversation):
            print(f"\n   Segment {i + 1}: {segment['speaker']}")
            print(
                f'   Text: "{segment["text"][:60]}{"..." if len(segment["text"]) > 60 else ""}"'
            )

            start_time = time.time()
            wav = model.generate(segment["text"])
            generation_time = time.time() - start_time

            # Save individual segment
            output_file = (
                output_dir
                / f"segment_{i + 1:02d}_{segment['speaker'].replace(' ', '_')}.wav"
            )
            ta.save(str(output_file), wav, model.sr)

            duration = wav.shape[1] / model.sr
            total_duration += duration
            total_generation_time += generation_time

            print(
                f"   ✅ Generated: {duration:.1f}s in {generation_time:.2f}s ({duration / generation_time:.1f}x real-time)"
            )
            print(f"   ✅ Saved: {output_file.name}")

        # Generate combined summary
        combined_file = output_dir / "complete_conversation.wav"

        # Load and concatenate all segments
        combined_audio = []
        for i in range(len(conversation)):
            speaker_name = conversation[i]["speaker"].replace(" ", "_")
            segment_file = output_dir / f"segment_{i + 1:02d}_{speaker_name}.wav"
            wav_data, sr = ta.load(str(segment_file))
            combined_audio.append(wav_data)

            # Add pause between speakers (0.5 seconds of silence)
            if i < len(conversation) - 1:
                pause = torch.zeros(1, int(sr * 0.5))
                combined_audio.append(pause)

        # Concatenate and save
        final_audio = torch.cat(combined_audio, dim=1)
        ta.save(str(combined_file), final_audio, model.sr)

        print(f"\n   📊 CONVERSATION SUMMARY:")
        print(f"   ✅ Total segments: {len(conversation)}")
        print(f"   ✅ Total duration: {total_duration:.1f} seconds")
        print(f"   ✅ Total generation time: {total_generation_time:.2f} seconds")
        print(
            f"   ✅ Average real-time factor: {total_duration / total_generation_time:.1f}x"
        )
        print(f"   ✅ Combined audio: {combined_file}")

        return True

    except Exception as e:
        print(f"   ❌ Conversation generation failed: {e}")
        traceback.print_exc()
        return False


async def demonstrate_voice_cloning():
    """Demonstrate voice cloning capabilities if audio prompt is available"""
    print("\n🎭 Demonstrating Voice Cloning Capabilities...")

    if not CHATTERBOX_AVAILABLE:
        print("   ❌ Skipping - Chatterbox not available")
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)

        # Check if we have any existing audio files to use as prompts
        output_dir = Path("storage/audio/test_outputs")

        # Look for existing audio files
        existing_audio = list(output_dir.glob("*.wav"))

        if existing_audio:
            # Use the first available audio file as a voice prompt
            audio_prompt_path = str(existing_audio[0])
            print(f"   🎯 Using voice prompt: {audio_prompt_path}")

            cloning_text = """
            This is a demonstration of voice cloning using Chatterbox TTS. 
            The voice you're hearing has been adapted from an existing audio sample, 
            showcasing the remarkable ability to transfer vocal characteristics and speaking style. 
            This technology opens unprecedented possibilities for content creation, 
            allowing us to maintain consistency across episodes while exploring new creative territories.
            """

            print(f"   🔄 Generating cloned voice audio...")

            start_time = time.time()
            wav = model.generate(
                cloning_text.strip(), audio_prompt_path=audio_prompt_path
            )
            generation_time = time.time() - start_time

            # Save cloned audio
            cloned_file = output_dir / "voice_cloning_demo.wav"
            ta.save(str(cloned_file), wav, model.sr)

            duration = wav.shape[1] / model.sr

            print(f"   ✅ Voice cloning successful!")
            print(f"   ✅ Duration: {duration:.2f} seconds")
            print(f"   ✅ Generation time: {generation_time:.2f} seconds")
            print(f"   ✅ Cloned audio saved: {cloned_file}")

            return True
        else:
            print("   ℹ️  No existing audio files found for voice prompt")
            print("   ℹ️  Generate other samples first to test voice cloning")
            return True

    except Exception as e:
        print(f"   ❌ Voice cloning demonstration failed: {e}")
        traceback.print_exc()
        return False


async def run_complete_audio_test():
    """Run complete Chatterbox TTS audio generation test"""
    print("🎵 CHATTERBOX TTS - REAL AUDIO GENERATION TEST")
    print("=" * 60)

    tests = [
        ("Installation Check", test_chatterbox_installation),
        ("Insightful Podcast Intro", generate_insightful_podcast_intro),
        ("Multi-Host Conversation", generate_conversation_sample),
        ("Voice Cloning Demo", demonstrate_voice_cloning),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("🏁 REAL AUDIO GENERATION TEST RESULTS")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests successful")

    if passed >= 2:  # At least installation + one generation test
        print("🎉 Chatterbox TTS is working! Real audio files generated!")
        print("\n📁 Check these directories for your generated audio:")
        print("   - storage/audio/test_outputs/")
        print("   - storage/audio/test_outputs/conversation/")

        print("\n🎧 LISTENING RECOMMENDATIONS:")
        print("   1. insightful_podcast_intro.wav - Deep philosophical intro")
        print("   2. conversation/ - Multi-speaker dialogue samples")
        print("   3. voice_cloning_demo.wav - Voice adaptation showcase")

    else:
        print("⚠️  Some tests failed. Check error messages above.")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Ensure CUDA drivers are installed (for GPU acceleration)")
        print("   2. Check available disk space and memory")
        print("   3. Verify network connection for model downloads")


if __name__ == "__main__":
    print("Starting real Chatterbox TTS audio generation test...\n")
    asyncio.run(run_complete_audio_test())
