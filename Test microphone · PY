"""
Microphone Testing Utility
Test your microphone and audio settings before running the speech agent.
"""

import speech_recognition as sr
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_microphone_detection():
    """Test if any microphones are detected."""
    logger.info("=" * 60)
    logger.info("🎤 MICROPHONE DETECTION TEST")
    logger.info("=" * 60 + "\n")
    
    try:
        mic_list = sr.Microphone.list_microphone_indexes()
        
        if not mic_list:
            logger.warning("❌ No microphones detected!")
            logger.info("Please check your audio device connections.")
            return False
        
        logger.info("✅ Microphones detected:\n")
        for index in mic_list:
            try:
                mic = sr.Microphone(device_index=index)
                logger.info(f"   [{index}] {sr.Microphone().get_pyaudio_instance().get_device_info_by_index(index)['name']}")
            except Exception as e:
                logger.info(f"   [{index}] Microphone (couldn't get name: {e})")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error detecting microphones: {e}")
        return False


def test_microphone_audio_levels():
    """Test microphone audio input levels."""
    logger.info("\n" + "=" * 60)
    logger.info("🔊 AUDIO LEVELS TEST")
    logger.info("=" * 60)
    logger.info("Listening for 5 seconds... Please make various sounds:\n")
    logger.info("  1. First 2 seconds - Stay silent (baseline)")
    logger.info("  2. Next 2 seconds - Speak normally")
    logger.info("  3. Last 1 second - Make loud sounds\n")
    
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logger.info("Calibrating microphone (3 seconds)...")
            recognizer.adjust_for_ambient_noise(source, duration=3)
            logger.info("✅ Calibration complete\n")
            
            logger.info("▶️  Recording audio levels...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            logger.info("⏹️  Recording complete\n")
            
            # Analyze audio
            audio_array = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
            
            if len(audio_array) == 0:
                logger.warning("❌ No audio data captured")
                return False
            
            # Calculate statistics
            rms = np.sqrt(np.mean(audio_array ** 2))
            db = 20 * np.log10(rms / 32768) if rms > 0 else 0
            peak = np.max(np.abs(audio_array))
            peak_db = 20 * np.log10(peak / 32768) if peak > 0 else 0
            
            logger.info("📊 Audio Analysis:")
            logger.info(f"  RMS Level: {rms:.0f} ({db:.1f} dB)")
            logger.info(f"  Peak Level: {peak:.0f} ({peak_db:.1f} dB)")
            logger.info(f"  Audio Duration: {len(audio_array) / audio.sample_rate:.2f} seconds")
            logger.info(f"  Sample Rate: {audio.sample_rate} Hz\n")
            
            # Provide recommendations
            logger.info("📋 Recommendations:")
            if db < 20:
                logger.info("  ⚠️  Audio is very quiet. Check microphone volume or positioning.")
            elif db > 60:
                logger.info("  ⚠️  Audio is very loud. Watch for clipping/distortion.")
            else:
                logger.info("  ✅ Audio levels look good!")
            
            return True
    
    except sr.UnknownValueError:
        logger.info("✅ Audio captured but couldn't recognize speech (that's okay for this test)")
        return True
    except sr.RequestError as e:
        logger.error(f"❌ Microphone error: {e}")
        return False


def test_speech_recognition():
    """Test speech recognition capability."""
    logger.info("\n" + "=" * 60)
    logger.info("🗣️  SPEECH RECOGNITION TEST")
    logger.info("=" * 60)
    logger.info("Listening for speech (5 seconds)... Say something!\n")
    
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info("▶️  Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            logger.info("🔄 Recognizing speech...")
            text = recognizer.recognize_google(audio)
            
            logger.info(f"✅ Recognized: '{text}'")
            logger.info("✅ Google Speech-to-Text works!\n")
            return True
    
    except sr.UnknownValueError:
        logger.warning("❌ Could not understand the audio. Speak more clearly.")
        return False
    except sr.RequestError as e:
        logger.error(f"❌ Speech recognition error: {e}")
        logger.info("   Make sure you have an internet connection.")
        return False
    except sr.WaitTimeoutError:
        logger.warning("⏱️  No speech detected. Try speaking louder.")
        return False


def test_audio_threshold_settings():
    """Test recommended threshold settings."""
    logger.info("\n" + "=" * 60)
    logger.info("⚙️  AUDIO THRESHOLD RECOMMENDATIONS")
    logger.info("=" * 60 + "\n")
    
    recognizer = sr.Recognizer()
    
    logger.info("Testing audio in your environment...")
    logger.info("(5 seconds of ambient sound)\n")
    
    try:
        with sr.Microphone() as source:
            audio_samples = []
            
            for i in range(5):
                try:
                    logger.info(f"Sample {i + 1}/5...")
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=1)
                    
                    audio_array = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_array ** 2))
                    db = 20 * np.log10(rms / 32768) if rms > 0 else 0
                    audio_samples.append(db)
                
                except:
                    pass
            
            if audio_samples:
                avg_db = np.mean(audio_samples)
                max_db = np.max(audio_samples)
                min_db = np.min(audio_samples)
                
                logger.info(f"\n📊 Ambient Audio Analysis:")
                logger.info(f"  Average Level: {avg_db:.1f} dB")
                logger.info(f"  Max Level: {max_db:.1f} dB")
                logger.info(f"  Min Level: {min_db:.1f} dB\n")
                
                logger.info("📋 Recommended .env Settings:")
                
                # Calculate thresholds based on environment
                silence_threshold = max(min_db - 10, 20)
                noise_threshold = avg_db + 10
                loud_threshold = max_db + 10
                
                logger.info(f"  SILENCE_THRESHOLD={int(silence_threshold)}")
                logger.info(f"  NOISE_THRESHOLD={int(noise_threshold)}")
                logger.info(f"  LOUD_THRESHOLD={int(loud_threshold)}")
                logger.info(f"  ENERGY_THRESHOLD=4000 (default)\n")
                
                logger.info("Copy these values to your .env file for optimal performance.")
                
                return True
            else:
                logger.warning("Could not collect audio samples.")
                return False
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 12 + "🎤 SPEECH AGENT MICROPHONE TEST 🎤" + " " * 12 + "║")
    logger.info("╚" + "=" * 58 + "╝\n")
    
    results = {
        "Microphone Detection": test_microphone_detection(),
        "Audio Levels": test_microphone_audio_levels(),
        "Speech Recognition": test_speech_recognition(),
        "Audio Threshold Settings": test_audio_threshold_settings(),
    }
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ TEST SUMMARY")
    logger.info("=" * 60 + "\n")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:.<45} {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("🎉 All tests passed! You're ready to use the speech agent.")
        logger.info("Run: python speech_agent.py")
    else:
        logger.info("⚠️  Some tests failed. Check the recommendations above.")
        logger.info("Common fixes:")
        logger.info("  1. Check microphone connections")
        logger.info("  2. Adjust system volume")
        logger.info("  3. Check internet connection")
        logger.info("  4. Update audio drivers")
    logger.info("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
