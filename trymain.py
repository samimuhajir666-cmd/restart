import os
import sys
import io
import time
import numpy as np
from collections import deque
import speech_recognition as sr
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WhisperSpeechAgent:
    """
    Advanced speech-to-text AI agent using OpenAI's Whisper API.
    Superior accuracy with multiple language support and noise robustness.
    """
    
    def __init__(self):
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Audio level thresholds (dB)
        self.silence_threshold = float(os.getenv('SILENCE_THRESHOLD', '30'))
        self.loud_threshold = float(os.getenv('LOUD_THRESHOLD', '70'))
        self.noise_threshold = float(os.getenv('NOISE_THRESHOLD', '40'))
        
        # Adaptive noise detection
        self.noise_buffer = deque(maxlen=10)
        self.is_listening = False
        self.should_exit = False
        self.transcribed_text = []
        self.confidence_scores = []
        
        # Configuration
        self.language = os.getenv('LANGUAGE', 'en')
        self.model = os.getenv('WHISPER_MODEL', 'base')  # tiny, base, small, medium, large
        
        # Speech recognition settings
        self.recognizer.energy_threshold = float(os.getenv('ENERGY_THRESHOLD', '4000'))
        self.recognizer.dynamic_energy_threshold = True
        
        logger.info("🤖 Whisper Speech Agent initialized")
        logger.info(f"Model: {self.model} | Language: {self.language}")
        logger.info(f"Silence threshold: {self.silence_threshold} dB")
        logger.info(f"Loud threshold: {self.loud_threshold} dB")
        logger.info(f"Noise threshold: {self.noise_threshold} dB\n")
    
    def calculate_audio_level(self, audio_data):
        """Calculate audio level in dB from audio data."""
        try:
            audio_array = np.frombuffer(audio_data.get_wav_data(), dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_array ** 2))
            
            # Convert to dB
            if rms > 0:
                db = 20 * np.log10(rms / 32768)
                return max(0, db)
            return 0
        except Exception as e:
            logger.warning(f"Error calculating audio level: {e}")
            return 0
    
    def is_background_noise(self, audio_level):
        """Determine if current audio is background noise."""
        self.noise_buffer.append(audio_level)
        
        if len(self.noise_buffer) >= 5:
            avg_noise = np.mean(list(self.noise_buffer))
            return audio_level < self.noise_threshold and audio_level < (avg_noise + 5)
        
        return audio_level < self.noise_threshold
    
    def is_too_loud(self, audio_level):
        """Check if audio is too loud (void/clipping detection)."""
        is_loud = audio_level > self.loud_threshold
        if is_loud:
            logger.warning(f"⚠️  LOUD SOUND DETECTED: {audio_level:.1f} dB - Skipping segment")
        return is_loud
    
    def is_silence(self, audio_level):
        """Check if audio is silence."""
        return audio_level < self.silence_threshold
    
    def transcribe_with_whisper(self, audio):
        """Transcribe audio using OpenAI Whisper API."""
        try:
            # Convert audio to WAV format
            wav_data = audio.get_wav_data()
            audio_file = io.BytesIO(wav_data)
            audio_file.name = "audio.wav"
            
            logger.info("🔄 Sending to Whisper API...")
            
            # Call Whisper API
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=self.language,
                temperature=0.0  # More deterministic results
            )
            
            text = transcript.text.strip()
            
            # Extract confidence if available
            confidence = getattr(transcript, 'confidence', None)
            if confidence:
                self.confidence_scores.append(confidence)
            
            return text, confidence
        
        except Exception as e:
            logger.error(f"❌ Whisper API error: {e}")
            return None, 0
    
    def detect_language(self, audio):
        """Detect language of audio."""
        try:
            audio_file = io.BytesIO(audio.get_wav_data())
            audio_file.name = "audio.wav"
            
            detection = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
            return getattr(detection, 'language', 'unknown')
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None
    
    def listen_and_transcribe(self):
        """Main listening loop with Whisper transcription."""
        logger.info("🎤 Agent is ready to listen. Speak naturally...")
        logger.info("Say 'exit' or 'quit' to stop the agent\n")
        
        try:
            with self.microphone as source:
                logger.info("Calibrating for ambient noise (3 seconds)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
                logger.info("Calibration complete. Listening...\n")
                
                while not self.should_exit:
                    try:
                        self.is_listening = True
                        
                        logger.info("▶️  Listening...")
                        audio = self.recognizer.listen(
                            source,
                            timeout=5,
                            phrase_time_limit=15
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
                        
                        # Process with Whisper
                        logger.info(f"📊 Audio level: {audio_level:.1f} dB - Processing...")
                        self.process_audio(audio)
                        
                    except sr.UnknownValueError:
                        logger.info("❌ Could not capture audio. Please speak clearly.\n")
                    except sr.RequestError as e:
                        logger.error(f"❌ Microphone error: {e}\n")
                    except sr.WaitTimeoutError:
                        logger.info("⏱️  No speech detected within timeout\n")
                    
                    time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Fatal error in listening loop: {e}")
        finally:
            self.is_listening = False
            logger.info("\n🛑 Agent stopped")
    
    def process_audio(self, audio):
        """Process and transcribe audio with Whisper."""
        text, confidence = self.transcribe_with_whisper(audio)
        
        if not text:
            logger.info("❌ Failed to transcribe audio\n")
            return
        
        # Check if user wants to exit
        if text.lower() in ['exit', 'quit', 'bye', 'goodbye', 'stop']:
            logger.info(f"You said: '{text}'")
            self.should_exit = True
            return
        
        # Display result
        confidence_str = f" (Confidence: {confidence:.1%})" if confidence else ""
        logger.info(f"✅ You said: '{text}'{confidence_str}")
        self.transcribed_text.append(text)
        
        # Process with AI
        self.process_transcribed_text(text)
        print()  # New line for readability
    
    def process_transcribed_text(self, text):
        """
        Process transcribed text with AI logic.
        Can be extended with LLM integration.
        """
        text_lower = text.lower()
        
        # Simple intent recognition
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            response = "👋 Hello! How can I assist you?"
        elif any(word in text_lower for word in ['time', 'what time']):
            current_time = time.strftime("%H:%M:%S")
            response = f"⏰ Current time is {current_time}"
        elif any(word in text_lower for word in ['thank', 'thanks']):
            response = "😊 You're welcome!"
        elif any(word in text_lower for word in ['help', 'assist']):
            response = "🆘 I can help with time, greetings, and basic tasks. What do you need?"
        else:
            response = "🤔 I understood you. How can I help?"
        
        logger.info(f"Agent: {response}")
    
    def get_session_summary(self):
        """Get detailed session summary."""
        summary = {
            'total_transcriptions': len(self.transcribed_text),
            'average_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'transcriptions': self.transcribed_text
        }
        return summary
    
    def start(self):
        """Start the agent."""
        try:
            self.listen_and_transcribe()
        except KeyboardInterrupt:
            logger.info("\n\n⏹️  Agent interrupted by user")
            self.should_exit = True
    
    def stop(self):
        """Stop the agent."""
        self.should_exit = True
        logger.info("Agent stopping...")


def main():
    """Main entry point."""
    try:
        agent = WhisperSpeechAgent()
        agent.start()
        
        # Print summary
        summary = agent.get_session_summary()
        logger.info(f"\n📋 Session Summary:")
        logger.info(f"Total Transcriptions: {summary['total_transcriptions']}")
        logger.info(f"Average Confidence: {summary['average_confidence']:.1%}")
        if summary['transcriptions']:
            logger.info(f"Transcribed Text: {' '.join(summary['transcriptions'])}")
    
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        logger.info("Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
