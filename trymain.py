#!/usr/bin/env python3
"""
🎤 Real-Time Speech-to-Text AI Agent
A professional-grade speech recognition agent that listens intelligently like humans do,
with adaptive noise detection and audio level management.

Installation:
    pip install SpeechRecognition pyaudio pydub numpy

Usage:
    python speech_agent.py

Author: AI Speech Agent
License: MIT
"""

import os
import sys
import time
import logging
import numpy as np
from collections import deque
import speech_recognition as sr

# ============================================================================
# CONFIGURATION - Adjust these values for your environment
# ============================================================================

# Audio Thresholds (in dB)
SILENCE_THRESHOLD = 30          # Below this = silence (ignored)
LOUD_THRESHOLD = 70             # Above this = distortion (skipped)
NOISE_THRESHOLD = 40            # Background noise detection level
ENERGY_THRESHOLD = 4000         # Speech sensitivity (0-4000, higher = less sensitive)

# Timeout Settings (in seconds)
LISTEN_TIMEOUT = 5              # How long to wait for speech
PHRASE_TIME_LIMIT = 15          # Max duration of single phrase
LISTEN_CYCLE_PAUSE = 0.5        # Pause between listening cycles

# Agent Configuration
AGENT_NAME = "VoiceAssistant"
LANGUAGE = "en-US"

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# REAL-TIME SPEECH AGENT CLASS
# ============================================================================

class RealTimeSpeechAgent:
    """
    Real-time speech-to-text AI agent with intelligent noise detection.
    Listens like humans do - context-aware, adaptive to background noise.
    """
    
    def __init__(self):
        """Initialize the speech agent."""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Audio level thresholds
        self.silence_threshold = SILENCE_THRESHOLD
        self.loud_threshold = LOUD_THRESHOLD
        self.noise_threshold = NOISE_THRESHOLD
        
        # Adaptive noise detection
        self.noise_buffer = deque(maxlen=10)  # Last 10 readings
        self.is_listening = False
        self.should_exit = False
        self.transcribed_text = []
        
        # Speech recognition settings
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True
        
        logger.info(f"🤖 {AGENT_NAME} initialized")
        logger.info(f"Silence threshold: {self.silence_threshold} dB")
        logger.info(f"Loud threshold: {self.loud_threshold} dB")
        logger.info(f"Noise threshold: {self.noise_threshold} dB\n")
    
    def calculate_audio_level(self, audio_data):
        """
        Calculate audio level in dB from audio data.
        
        Args:
            audio_data: AudioData object from speech_recognition
            
        Returns:
            float: Audio level in dB (0-100)
        """
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16)
            
            # Avoid log of zero
            if len(audio_array) == 0:
                return 0
            
            # Calculate RMS (Root Mean Square)
            rms = np.sqrt(np.mean(audio_array ** 2))
            
            # Convert to dB (20 * log10(RMS / reference))
            if rms > 0:
                db = 20 * np.log10(rms / 32768)
                return max(0, db)
            return 0
        except Exception as e:
            logger.warning(f"Error calculating audio level: {e}")
            return 0
    
    def is_background_noise(self, audio_level):
        """
        Determine if current audio is background noise using adaptive detection.
        
        Args:
            audio_level: Current audio level in dB
            
        Returns:
            bool: True if audio is background noise
        """
        # Add to buffer for adaptive detection
        self.noise_buffer.append(audio_level)
        
        # If enough samples, calculate average noise level
        if len(self.noise_buffer) >= 5:
            avg_noise = np.mean(list(self.noise_buffer))
            return audio_level < self.noise_threshold and audio_level < (avg_noise + 5)
        
        return audio_level < self.noise_threshold
    
    def is_too_loud(self, audio_level):
        """
        Check if audio is too loud (void/clipping detection).
        
        Args:
            audio_level: Current audio level in dB
            
        Returns:
            bool: True if audio is too loud
        """
        is_loud = audio_level > self.loud_threshold
        if is_loud:
            logger.warning(f"⚠️  LOUD SOUND DETECTED: {audio_level:.1f} dB - Skipping this segment")
        return is_loud
    
    def is_silence(self, audio_level):
        """
        Check if audio is silence.
        
        Args:
            audio_level: Current audio level in dB
            
        Returns:
            bool: True if audio is silence
        """
        return audio_level < self.silence_threshold
    
    def listen_and_transcribe(self):
        """Main listening loop with intelligent audio analysis."""
        logger.info("🎤 Agent is ready to listen. Speak naturally...")
        logger.info("Say 'exit' or 'quit' to stop the agent\n")
        
        try:
            with self.microphone as source:
                # Calibrate microphone for ambient noise
                logger.info("Calibrating for ambient noise (3 seconds)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
                logger.info("✅ Calibration complete. Listening...\n")
                
                while not self.should_exit:
                    try:
                        self.is_listening = True
                        
                        # Listen with timeout
                        logger.info("▶️  Listening...")
                        audio = self.recognizer.listen(
                            source,
                            timeout=LISTEN_TIMEOUT,
                            phrase_time_limit=PHRASE_TIME_LIMIT
                        )
                        
                        # Calculate audio level
                        audio_level = self.calculate_audio_level(audio)
                        
                        # Intelligent audio filtering
                        if self.is_too_loud(audio_level):
                            logger.info("🔇 Discarding loud/distorted audio\n")
                            continue
                        
                        if self.is_silence(audio_level):
                            logger.info("🤐 Silence detected\n")
                            continue
                        
                        if self.is_background_noise(audio_level):
                            logger.info(f"🌫️  Background noise detected ({audio_level:.1f} dB) - Filtering...\n")
                            continue
                        
                        # Process audio
                        logger.info(f"📊 Audio level: {audio_level:.1f} dB - Processing...")
                        self.process_audio(audio)
                        
                    except sr.UnknownValueError:
                        logger.info("❌ Could not understand audio. Please speak clearly.\n")
                    except sr.RequestError as e:
                        logger.error(f"❌ API error: {e}\n")
                    except sr.WaitTimeoutError:
                        logger.info("⏱️  No speech detected within timeout\n")
                    
                    time.sleep(LISTEN_CYCLE_PAUSE)  # Brief pause between cycles
        
        except Exception as e:
            logger.error(f"Fatal error in listening loop: {e}")
        finally:
            self.is_listening = False
            logger.info("\n🛑 Agent stopped")
    
    def process_audio(self, audio):
        """
        Process and transcribe audio.
        
        Args:
            audio: AudioData object from speech_recognition
        """
        try:
            # Use Google Speech Recognition (free option)
            logger.info("🔄 Transcribing...")
            text = self.recognizer.recognize_google(audio)
            
            # Check if user wants to exit
            if text.lower() in ['exit', 'quit', 'bye', 'goodbye', 'stop']:
                logger.info(f"You said: '{text}'")
                self.should_exit = True
                return
            
            logger.info(f"✅ You said: '{text}'")
            self.transcribed_text.append(text)
            
            # Process text with AI agent logic
            self.process_transcribed_text(text)
            print()  # New line for readability
            
        except sr.UnknownValueError:
            logger.info("❌ Speech too unclear\n")
        except sr.RequestError as e:
            logger.error(f"❌ Speech recognition error: {e}\n")
    
    def process_transcribed_text(self, text):
        """
        Process transcribed text with AI agent logic.
        
        This is where you add custom AI behavior.
        Override this method for custom logic.
        
        Args:
            text: Transcribed text from user
        """
        # Simple intent detection
        text_lower = text.lower()
        
        # ====== CUSTOMIZE AI RESPONSES HERE ======
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            response = "👋 Hello! How can I help you?"
        
        elif any(word in text_lower for word in ['time', 'what time']):
            current_time = time.strftime("%H:%M:%S")
            response = f"⏰ Current time is {current_time}"
        
        elif any(word in text_lower for word in ['thank', 'thanks']):
            response = "😊 You're welcome!"
        
        elif any(word in text_lower for word in ['help', 'assist']):
            response = "🆘 I can help with time, greetings, and basic tasks. What do you need?"
        
        else:
            response = "🤔 I heard you. What would you like me to do?"
        
        # ==========================================
        
        logger.info(f"Agent: {response}")
    
    def get_transcription_summary(self):
        """Get summary of all transcribed text."""
        if not self.transcribed_text:
            return "No transcriptions recorded"
        return " ".join(self.transcribed_text)
    
    def start(self):
        """Start the speech agent."""
        try:
            self.listen_and_transcribe()
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Agent interrupted by user")
            self.should_exit = True
    
    def stop(self):
        """Stop the speech agent."""
        self.should_exit = True
        logger.info("Agent stopping...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    agent = RealTimeSpeechAgent()
    
    try:
        agent.start()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        # Print summary
        summary = agent.get_transcription_summary()
        logger.info(f"\n📋 Session Summary:\n{summary}")


if __name__ == "__main__":
    main()
