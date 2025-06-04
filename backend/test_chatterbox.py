#!/usr/bin/env python3
"""
Test script for Chatterbox TTS integration

This script tests the Chatterbox TTS functionality to ensure our migration
from ElevenLabs to Chatterbox is working correctly.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.chatterbox_service import chatterbox_service
from app.services.voice_agent import voice_agent


async def test_basic_chatterbox():
    """Test basic Chatterbox TTS functionality"""
    print("🎤 Testing Basic Chatterbox TTS...")

    try:
        # Test service availability
        available = chatterbox_service.is_available()
        print(f"   ✓ Service available: {available}")

        # Test connection
        connection_test = await chatterbox_service.test_connection()
        print(f"   ✓ Connection test: {connection_test['status']}")
        print(f"   ✓ Device: {connection_test.get('device', 'unknown')}")
        print(f"   ✓ Sample rate: {connection_test.get('sample_rate', 'unknown')}")

        return connection_test["status"] == "success"

    except Exception as e:
        print(f"   ❌ Basic test failed: {e}")
        return False


async def test_text_to_speech():
    """Test text-to-speech conversion"""
    print("\n🗣️ Testing Text-to-Speech Conversion...")

    test_text = "Hello, this is a test of Chatterbox text-to-speech. The migration from ElevenLabs appears to be successful!"

    try:
        start_time = time.time()

        # Generate speech
        tts_response = await chatterbox_service.convert_text_to_speech(
            text=test_text, voice_id="alex"
        )

        processing_time = time.time() - start_time

        if tts_response.success:
            print(f"   ✓ TTS generation successful")
            print(f"   ✓ Voice ID: {tts_response.voice_id}")
            print(f"   ✓ Duration: {tts_response.duration:.2f} seconds")
            print(f"   ✓ Audio format: {tts_response.audio_format}")
            print(f"   ✓ Sample rate: {tts_response.sample_rate}")
            print(f"   ✓ Processing time: {processing_time:.2f} seconds")
            print(f"   ✓ Audio data size: {len(tts_response.audio_data)} bytes")

            # Save audio file for testing
            output_dir = Path("storage/audio/test_outputs")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "test_basic_tts.wav"
            with open(output_file, "wb") as f:
                f.write(tts_response.audio_data)
            print(f"   ✓ Audio saved to: {output_file}")

            return True
        else:
            print(f"   ❌ TTS generation failed: {tts_response.error_message}")
            return False

    except Exception as e:
        print(f"   ❌ TTS test failed: {e}")
        return False


async def test_voice_profiles():
    """Test voice profile functionality"""
    print("\n👥 Testing Voice Profiles...")

    try:
        # Get available voices
        voices = chatterbox_service.get_available_voices()
        print(f"   ✓ Available voices: {len(voices)}")

        for voice in voices:
            print(f"     - {voice['voice_id']}: {voice['name']} ({voice['style']})")

        # Test podcast voice profiles
        podcast_voices = chatterbox_service.get_podcast_voices()
        print(f"   ✓ Podcast voice profiles: {len(podcast_voices)}")

        for speaker_id, profile in podcast_voices.items():
            print(f"     - {speaker_id}: {profile['name']} - {profile['role']}")

        return True

    except Exception as e:
        print(f"   ❌ Voice profiles test failed: {e}")
        return False


async def test_voice_agent():
    """Test Voice Agent functionality"""
    print("\n🤖 Testing Voice Agent...")

    try:
        # Test availability
        available = voice_agent.is_available()
        print(f"   ✓ Voice Agent available: {available}")

        # Test health check
        health = await voice_agent.health_check()
        print(f"   ✓ Health status: {health['status']}")

        # Test single segment generation
        test_text = "Welcome to VoiceFlow Studio, powered by Chatterbox TTS!"
        segment = await voice_agent.generate_single_segment(
            text=test_text, speaker_id="host1"
        )

        if segment.success:
            print(f"   ✓ Single segment generation successful")
            print(f"   ✓ Speaker: {segment.speaker_id}")
            print(f"   ✓ Voice: {segment.voice_id}")
            print(f"   ✓ Duration: {segment.duration:.2f} seconds")
            print(
                f"   ✓ Audio data size: {len(segment.audio_data) if segment.audio_data else 0} bytes"
            )

            # Save test audio
            if segment.audio_data:
                output_dir = Path("storage/audio/test_outputs")
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / f"test_voice_agent_{segment.speaker_id}.wav"
                with open(output_file, "wb") as f:
                    f.write(segment.audio_data)
                print(f"   ✓ Audio saved to: {output_file}")

            return True
        else:
            print(f"   ❌ Single segment generation failed: {segment.error_message}")
            return False

    except Exception as e:
        print(f"   ❌ Voice Agent test failed: {e}")
        return False


async def test_podcast_segment_generation():
    """Test podcast segment generation with multiple speakers"""
    print("\n🎙️ Testing Podcast Segment Generation...")

    script_segments = [
        {
            "text": "Welcome to today's episode of Tech Talk. I'm Alex, your host.",
            "speaker": "host1",
            "type": "intro",
        },
        {
            "text": "And I'm Sarah, co-hosting today's discussion about artificial intelligence.",
            "speaker": "host2",
            "type": "intro",
        },
        {
            "text": "Today we're exploring the fascinating world of open-source text-to-speech technology.",
            "speaker": "host1",
            "type": "dialogue",
        },
        {
            "text": "That's right, Alex. We've recently migrated from ElevenLabs to Chatterbox TTS, and the results are impressive.",
            "speaker": "host2",
            "type": "dialogue",
        },
    ]

    try:
        start_time = time.time()

        # Generate voice segments
        result = await voice_agent.generate_voice_segments(
            script_segments=script_segments, include_cost_estimate=True
        )

        processing_time = time.time() - start_time

        if result.success:
            print(f"   ✓ Podcast generation successful")
            print(f"   ✓ Total segments: {len(result.audio_segments)}")
            print(
                f"   ✓ Successful segments: {sum(1 for seg in result.audio_segments if seg['success'])}"
            )
            print(f"   ✓ Total duration: {result.total_duration:.2f} seconds")
            print(f"   ✓ Total cost: ${result.total_cost:.4f}")
            print(f"   ✓ Processing time: {processing_time:.2f} seconds")

            # Save individual segments
            output_dir = Path("storage/audio/test_outputs/podcast_segments")
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, segment in enumerate(result.audio_segments):
                if segment["success"] and segment["audio_data"]:
                    output_file = (
                        output_dir / f"segment_{i:02d}_{segment['speaker_id']}.wav"
                    )
                    with open(output_file, "wb") as f:
                        f.write(segment["audio_data"])
                    print(
                        f"     - Segment {i}: {segment['speaker_id']} - {segment['duration']:.1f}s -> {output_file.name}"
                    )

            return True
        else:
            print(f"   ❌ Podcast generation failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"   ❌ Podcast segment test failed: {e}")
        return False


async def test_cost_estimation():
    """Test cost estimation functionality"""
    print("\n💰 Testing Cost Estimation...")

    try:
        test_texts = [
            "Short test.",
            "This is a medium length test sentence to check cost estimation functionality.",
            "This is a much longer test paragraph that contains multiple sentences and should demonstrate how the cost estimation works with larger amounts of text. It includes various words and punctuation to simulate real podcast content.",
        ]

        for i, text in enumerate(test_texts):
            cost_info = chatterbox_service.estimate_cost(text)
            print(f"   ✓ Test {i + 1}:")
            print(f"     - Characters: {cost_info['character_count']}")
            print(
                f"     - Est. processing time: {cost_info['estimated_processing_time']:.2f}s"
            )
            print(f"     - Computational cost: {cost_info['computational_cost']}")
            print(f"     - API cost: ${cost_info['api_cost']:.4f}")

        return True

    except Exception as e:
        print(f"   ❌ Cost estimation test failed: {e}")
        return False


async def run_all_tests():
    """Run all Chatterbox tests"""
    print("🧪 CHATTERBOX TTS INTEGRATION TESTS")
    print("=" * 50)

    tests = [
        ("Basic Chatterbox", test_basic_chatterbox),
        ("Text-to-Speech", test_text_to_speech),
        ("Voice Profiles", test_voice_profiles),
        ("Voice Agent", test_voice_agent),
        ("Podcast Segments", test_podcast_segment_generation),
        ("Cost Estimation", test_cost_estimation),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All tests passed! Chatterbox migration successful!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    print(f"\n📁 Test outputs saved to: storage/audio/test_outputs/")


if __name__ == "__main__":
    print("Starting Chatterbox TTS integration tests...\n")
    asyncio.run(run_all_tests())
