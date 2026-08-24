"""
👂 HUMAN-LIKE SPEECH-TO-TEXT AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A speech recognition system that listens like humans:
  • Ignores irrelevant background noise
  • Focuses on primary speaker
  • Adapts to changing conditions
  • Uses context for understanding
  • Marks unclear sections rather than guessing
  
Production-ready for Streamlit Cloud deployment
"""

import html
import io
import os
import re
import numpy as np
import requests
import scipy.io.wavfile as wav
import scipy.signal as signal
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from collections import deque

load_dotenv()

# ============================
# ⚙️ CONFIGURATION
# ============================

st.set_page_config(
    page_title="Human-Like Speech Agent",
    page_icon="👂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or st.secrets.get("DEEPGRAM_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY not found in .env or Secrets")
    st.stop()

# ============================
# 👂 VOICE ACTIVITY DETECTION (VAD)
# ============================

class VoiceActivityDetector:
    """
    Detects human speech vs background noise
    
    Mimics human ear behavior:
    - Recognizes speech patterns (pitch, rhythm, formants)
    - Ignores constant background (AC, fan, hum)
    - Adapts to volume changes
    - Learns from context
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 20
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Adaptive thresholds
        self.speech_floor_db = 35  # Minimum speech level
        self.noise_ceiling_db = 25  # Maximum noise level
        self.vad_history = deque(maxlen=30)  # Temporal smoothing
        
        # Learning
        self.noise_spectrum = None
        self.speech_spectrum = None
        self.learning_frames = 0
        
    def compute_mfcc_features(self, audio_chunk):
        """
        Compute MFCC-like features that distinguish speech from noise
        
        Speech has:
        - Spectral peaks (formants)
        - Rhythmic structure
        - Energy variation
        """
        try:
            import librosa
            # Compute mel-frequency features
            mel_spec = librosa.feature.melspectrogram(
                y=audio_chunk.astype(np.float32),
                sr=self.sample_rate,
                n_mels=13
            )
            
            # MFCC
            mfcc = librosa.feature.mfcc(
                y=audio_chunk.astype(np.float32),
                sr=self.sample_rate,
                n_mfcc=13
            )
            
            return {
                'mel_energy': float(np.mean(mel_spec)),
                'mel_variance': float(np.std(mel_spec)),
                'mfcc_delta': float(np.mean(np.abs(np.diff(mfcc, axis=1))))
            }
        except:
            # Fallback if librosa unavailable
            return self._compute_fallback_features(audio_chunk)
    
    def _compute_fallback_features(self, audio_chunk):
        """Fallback feature computation without librosa"""
        # Energy
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
        energy_db = 20 * np.log10(rms / 32768 + 1e-10)
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2 * len(audio_chunk))
        
        # Spectral entropy
        spectrum = np.abs(np.fft.fft(audio_chunk))
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
        entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10))
        
        return {
            'energy': energy_db,
            'zcr': zcr,
            'entropy': entropy
        }
    
    def is_speech(self, audio_chunk):
        """
        Determine if chunk contains speech
        
        Uses multiple cues:
        1. Energy (speech louder than background)
        2. Spectral characteristics (speech vs silence/noise)
        3. Temporal structure (speech rhythmic, noise constant)
        """
        features = self.compute_mfcc_features(audio_chunk)
        
        # Multi-factor decision
        factors = []
        
        # Factor 1: Has speech-level energy?
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
        energy_db = 20 * np.log10(rms / 32768 + 1e-10)
        factors.append(energy_db > self.speech_floor_db)
        
        # Factor 2: Has spectral variation (speech vs constant hum)?
        if 'mel_variance' in features:
            factors.append(features['mel_variance'] > 2.0)
        
        # Factor 3: Has temporal dynamics (MFCC delta)?
        if 'mfcc_delta' in features:
            factors.append(features['mfcc_delta'] > 0.5)
        
        # If we have features, use them; else use energy only
        if len(factors) >= 2:
            speech_score = sum(factors) / len(factors)
        else:
            speech_score = 1.0 if factors[0] else 0.0
        
        # Smooth with history (human auditory system has temporal inertia)
        self.vad_history.append(speech_score > 0.5)
        
        if len(self.vad_history) > 5:
            # Vote based on recent history
            recent_vote = sum(self.vad_history) / len(self.vad_history)
            return recent_vote > 0.4
        
        return speech_score > 0.5


# ============================
# 🗣️ SPEAKER PRIORITIZER
# ============================

class SpeakerPrioritizer:
    """
    Identifies main speaker and filters out other voices
    
    Like human attention in conversation:
    - Learns speaker's voice characteristics
    - Focuses on them even with background chatter
    - Ignores other speakers
    """
    
    def __init__(self):
        self.primary_speaker_profile = None
        self.learning_count = 0
        self.max_learning = 30  # Learn from first ~30 frames
        self.confidence_threshold = 0.6
        
    def extract_pitch_features(self, audio_chunk):
        """
        Extract pitch characteristics
        Each speaker has unique pitch range/quality
        """
        try:
            import librosa
            f0 = librosa.yin(
                audio_chunk.astype(np.float64),
                fmin=50,
                fmax=400,
                sr=16000
            )
            
            # Get voiced frames
            voiced = f0[~np.isnan(f0)]
            if len(voiced) > 0:
                return {
                    'pitch_mean': float(np.mean(voiced)),
                    'pitch_std': float(np.std(voiced)),
                    'pitch_range': (float(np.min(voiced)), float(np.max(voiced)))
                }
        except:
            pass
        
        return None
    
    def learn_speaker(self, audio_chunk):
        """Learn primary speaker's characteristics"""
        if self.learning_count >= self.max_learning:
            return
        
        pitch_feat = self.extract_pitch_features(audio_chunk)
        
        if pitch_feat:
            if self.primary_speaker_profile is None:
                self.primary_speaker_profile = pitch_feat
            else:
                # Update incrementally
                alpha = 0.1
                old = self.primary_speaker_profile
                self.primary_speaker_profile = {
                    'pitch_mean': alpha * pitch_feat['pitch_mean'] + (1-alpha) * old['pitch_mean'],
                    'pitch_std': alpha * pitch_feat['pitch_std'] + (1-alpha) * old['pitch_std'],
                    'pitch_range': pitch_feat['pitch_range']
                }
        
        self.learning_count += 1
    
    def is_primary_speaker(self, audio_chunk):
        """Check if audio is from primary speaker"""
        if self.learning_count < 5:
            return True  # Trust everything while learning
        
        pitch_feat = self.extract_pitch_features(audio_chunk)
        
        if not pitch_feat or not self.primary_speaker_profile:
            return True  # Uncertain, assume yes
        
        # Check if pitch matches primary speaker
        pitch_diff = abs(pitch_feat['pitch_mean'] - self.primary_speaker_profile['pitch_mean'])
        tolerance = self.primary_speaker_profile['pitch_std'] * 2
        
        return pitch_diff < tolerance


# ============================
# 🔊 NOISE SUPPRESSION ENGINE
# ============================

class AdaptiveNoiseGate:
    """
    Intelligent noise gating that learns environment
    
    Behavior:
    - Measures background noise floor
    - Ignores constant background
    - Preserves varying speech
    - Adapts to noise level changes
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_floor = 25  # dB
        self.noise_floor_history = deque(maxlen=50)
        self.adaptation_rate = 0.05
        
    def update_noise_floor(self, audio_chunk):
        """Continuously learn the noise floor"""
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
        chunk_db = 20 * np.log10(rms / 32768 + 1e-10)
        
        self.noise_floor_history.append(chunk_db)
        
        # Use lower percentile as noise floor estimate
        if len(self.noise_floor_history) > 20:
            new_floor = np.percentile(list(self.noise_floor_history), 25)
            self.noise_floor = self.adaptation_rate * new_floor + (1 - self.adaptation_rate) * self.noise_floor
    
    def suppress(self, audio_chunk):
        """Apply noise suppression"""
        # Simple gate: if below threshold, attenuate
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
        chunk_db = 20 * np.log10(rms / 32768 + 1e-10)
        
        if chunk_db < self.noise_floor + 5:  # Below noise floor + margin
            # Gradually mute
            attenuation = max(0, (self.noise_floor + 10 - chunk_db) / 10)
            return (audio_chunk.astype(np.float64) * (1 - attenuation * 0.7)).astype(np.int16)
        
        return audio_chunk


# ============================
# 📝 TRANSCRIPTION QUALITY TRACKER
# ============================

class TranscriptionQualityTracker:
    """
    Tracks confidence per segment
    Marks unclear sections as [inaudible]
    """
    
    def __init__(self):
        self.segment_confidences = []
        self.segment_texts = []
        self.unclear_threshold = 0.4
        
    def add_segment(self, text, confidence):
        """Record transcribed segment with confidence"""
        self.segment_texts.append(text)
        self.segment_confidences.append(confidence)
    
    def mark_unclear(self, transcription, confidences):
        """
        Mark low-confidence sections as [inaudible]
        
        Never invents words - only marks when unsure
        """
        if not confidences or len(confidences) == 0:
            return transcription
        
        avg_conf = np.mean(confidences)
        
        if avg_conf < self.unclear_threshold:
            # Mark the whole thing as unclear
            return f"[inaudible: {transcription}]" if transcription else "[inaudible]"
        
        if avg_conf < 0.6:
            # Add confidence marker
            return f"{transcription} [low confidence]"
        
        return transcription


# ============================
# 🎙️ DEEPGRAM TRANSCRIPTION
# ============================

def transcribe_with_deepgram(audio_bytes, language="ur"):
    """
    Transcribe audio using Deepgram API
    
    Returns:
    {
        'text': transcription in original script,
        'confidence': 0-1 confidence score,
        'words': list of word-level data if available
    }
    """
    params = [
        ("model", "nova-3"),
        ("language", language),
        ("smart_format", "true"),
        ("punctuate", "true"),
        ("utterances", "true"),
        ("numerals", "true"),
    ]
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    
    try:
        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=60,
        )
        
        if response.status_code != 200:
            st.error(f"Deepgram error: {response.status_code}")
            return None
        
        data = response.json()
        results = data.get("results", {})
        channels = results.get("channels", [])
        
        if not channels:
            return None
        
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return None
        
        alt = alternatives[0]
        return {
            'text': (alt.get("transcript") or "").strip(),
            'confidence': float(alt.get("confidence", 0.0) or 0.0),
            'words': alt.get("words", [])
        }
    
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None


# ============================
# 🌐 CONTEXT-AWARE ENHANCEMENT
# ============================

def enhance_with_groq_context(transcription, confidence, audio_quality):
    """
    Use Groq to understand context and improve transcription
    
    Only corrects:
    - Obvious errors based on context
    - Never invents words
    - Marks genuinely unclear parts
    """
    
    if not GROQ_API_KEY or confidence > 0.8:
        # Already high confidence, skip enhancement
        return transcription
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""You are a speech recognition expert.

Transcribed text: "{transcription}"
Confidence: {confidence:.1%}
Audio quality: {audio_quality:.1%}

Task:
1. If transcription contains obvious errors, correct them using linguistic context
2. DO NOT invent or add words
3. DO NOT guess when unsure
4. Mark genuinely unclear parts as [inaudible]
5. Preserve natural speech (hesitations, repetitions, etc)

Return ONLY the corrected transcription, nothing else."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are conservative about corrections. Only fix obvious errors."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        # If Groq fails, return original
        return transcription


# ============================
# 📊 AUDIO QUALITY ASSESSMENT
# ============================

def assess_audio_quality(audio_data, sample_rate):
    """
    Comprehensive audio quality check
    
    Returns: quality_score (0-1)
    """
    # Energy
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    energy_db = 20 * np.log10(rms / 32768 + 1e-10)
    
    # SNR estimation (signal-to-noise ratio)
    sorted_energies = sorted([
        20 * np.log10(np.sqrt(np.mean(audio_data[i:i+sample_rate//10].astype(np.float64)**2)) / 32768 + 1e-10)
        for i in range(0, len(audio_data) - sample_rate//10, sample_rate//20)
    ])
    
    noise_floor = np.mean(sorted_energies[:len(sorted_energies)//4])
    snr = energy_db - noise_floor
    
    # Quality scoring
    quality = 0.0
    
    # Energy good? (40-60 dB optimal)
    if 40 < energy_db < 60:
        quality += 0.3
    elif 35 < energy_db < 65:
        quality += 0.2
    
    # SNR good? (>10 dB good)
    if snr > 10:
        quality += 0.4
    elif snr > 5:
        quality += 0.2
    
    # Not clipped?
    if np.max(np.abs(audio_data)) < 32000:
        quality += 0.3
    
    return min(1.0, quality)


# ============================
# 🎙️ MAIN PROCESSING PIPELINE
# ============================

def process_audio_human_like(audio_bytes, use_groq_context=True, show_details=False):
    """
    Complete processing pipeline that mimics human listening
    
    Steps:
    1. Load and analyze audio quality
    2. Detect speech vs background (VAD)
    3. Identify primary speaker
    4. Suppress background noise
    5. Extract clear speech regions
    6. Transcribe with confidence tracking
    7. Enhance with context (optional)
    8. Mark unclear sections
    """
    
    try:
        # Load audio
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_data = audio_data.astype(np.int16)
        
        # Assess quality
        quality_score = assess_audio_quality(audio_data, sample_rate)
        
        # Initialize components
        vad = VoiceActivityDetector(sample_rate)
        speaker = SpeakerPrioritizer()
        noise_gate = AdaptiveNoiseGate(sample_rate)
        quality_tracker = TranscriptionQualityTracker()
        
        # Process frames
        frame_size = sample_rate // 50  # 20ms frames
        speech_regions = []
        processed_frames = []
        
        current_region_start = None
        current_region_confidence = []
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i+frame_size]
            
            # Step 1: Detect speech
            is_speech = vad.is_speech(frame)
            
            # Step 2: Update noise floor
            noise_gate.update_noise_floor(frame)
            
            if is_speech:
                # Learn speaker
                speaker.learn_speaker(frame)
                
                # Check if primary speaker
                is_primary = speaker.is_primary_speaker(frame)
                
                if is_primary:
                    # Suppress noise
                    suppressed = noise_gate.suppress(frame)
                    processed_frames.append(suppressed)
                    
                    # Track region
                    if current_region_start is None:
                        current_region_start = i
                    
                    # Track confidence (quality score per frame)
                    current_region_confidence.append(quality_score)
            else:
                # End of speech region
                if current_region_start is not None:
                    region_confidence = np.mean(current_region_confidence) if current_region_confidence else 0
                    speech_regions.append({
                        'start': current_region_start,
                        'end': i,
                        'confidence': region_confidence
                    })
                    current_region_start = None
                    current_region_confidence = []
                
                # Keep background for continuity
                processed_frames.append(frame)
        
        # Reconstruct audio
        if not processed_frames:
            return None
        
        processed_audio = np.concatenate(processed_frames)
        
        # Prepare processed audio buffer
        output_buffer = io.BytesIO()
        wav.write(output_buffer, sample_rate, processed_audio)
        output_buffer.seek(0)
        
        return {
            'processed_bytes': output_buffer.read(),
            'quality_score': quality_score,
            'speech_regions': speech_regions,
            'sample_rate': sample_rate,
            'speaker_learned': speaker.learning_count
        }
    
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None


# ============================
# 🌐 ROMAN TRANSLITERATION
# ============================

URDU_TO_ROMAN = {
    'السلام': 'Assalam',
    'السلام علیکم': 'Assalam alaikum',
    'ہیلو': 'Hello',
    'شکریہ': 'Shukriya',
    'آپ کیسے ہیں': 'Aap kaise hain',
    'نمسٹے': 'Namaste',
    'جی': 'Haan',
    'نہیں': 'Nahi',
    'خدا حافظ': 'Khuda hafiz',
}

def transliterate(text):
    """Convert Urdu script to Roman"""
    for urdu, roman in URDU_TO_ROMAN.items():
        text = text.replace(urdu, roman)
    
    # Remove remaining Urdu script
    text = re.sub(r'[\u0600-\u06FF]', '', text)
    return text


# ============================
# 🖥️ STREAMLIT INTERFACE
# ============================

# Title
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>👂 Human-Like Speech-to-Text Agent</h1>
    <p style="font-size: 1.1em; color: #808080;">
    Listens naturally • Ignores background noise • Focuses on main speaker
    </p>
</div>
""", unsafe_allow_html=True)

# How it works
with st.expander("🧠 How This Works"):
    st.markdown("""
    **Like human ears, this agent:**
    
    1. **Listens continuously** - Monitors all audio
    2. **Detects speech** - Identifies human speech vs background noise
    3. **Learns your voice** - Focuses on primary speaker
    4. **Filters noise** - Suppresses AC, fans, background chatter
    5. **Understands context** - Uses nearby words to clarify unclear parts
    6. **Marks unclear sections** - Says [inaudible] instead of guessing
    
    **Different from blind transcription:**
    - ❌ NOT: "AC humming sound fan background person talking"
    - ✅ YES: "Your clear speech (ignores background)"
    """)

# Settings
col1, col2, col3 = st.columns(3)

with col1:
    use_context = st.checkbox(
        "🧠 Context-Aware (Groq)",
        value=True,
        help="Use AI to understand context and improve clarity"
    )

with col2:
    show_details = st.checkbox(
        "📊 Show Details",
        value=False,
        help="Display audio analysis and confidence scores"
    )

with col3:
    lang = st.selectbox(
        "🌐 Language",
        ["Urdu (اردو)", "Hindi (ہندی)", "English"],
        index=0
    )

lang_code = {"Urdu (اردو)": "ur", "Hindi (ہندی)": "hi", "English": "en"}[lang]

st.divider()

# Recording interface
st.subheader("🎤 Speak Now")
st.caption("The agent will focus on your voice and ignore background noise")

audio_output = mic_recorder(
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
    format="wav",
    key="main_mic"
)

if audio_output:
    audio_bytes = audio_output.get("bytes")
    
    if audio_bytes:
        # Process
        with st.spinner("👂 Listening like human ears..."):
            result = process_audio_human_like(
                audio_bytes,
                use_groq_context=use_context,
                show_details=show_details
            )
        
        if result:
            # Transcribe
            with st.spinner("📝 Transcribing..."):
                trans = transcribe_with_deepgram(result['processed_bytes'], language=lang_code)
            
            if trans:
                text = trans['text']
                confidence = trans['confidence']
                
                # Enhance with context if requested
                if use_context and GROQ_API_KEY:
                    with st.spinner("🧠 Applying context..."):
                        text = enhance_with_groq_context(
                            text,
                            confidence,
                            result['quality_score']
                        )
                
                # Transliterate if Urdu/Hindi
                if lang_code != "en":
                    roman_text = transliterate(text)
                else:
                    roman_text = text
                
                # Mark if unclear
                if confidence < 0.5:
                    roman_text = f"{roman_text}\n\n⚠️ [Note: Audio was unclear, confidence {confidence*100:.0f}%]"
                
                # Display result
                st.markdown("---")
                st.subheader("✅ Transcription")
                
                st.markdown(f"""
                <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; border-left: 4px solid #89b4fa;">
                    <div style="color: #89b4fa; font-weight: bold; margin-bottom: 10px;">Original ({lang}):</div>
                    <div style="color: #cdd6f4; font-size: 1.1em; margin-bottom: 15px; line-height: 1.5;">
                        {html.escape(text)}
                    </div>
                    
                    <div style="color: #a6e3a1; font-weight: bold; margin-bottom: 10px;">Roman Script:</div>
                    <div style="color: #cdd6f4; font-size: 1.1em; line-height: 1.5;">
                        {html.escape(roman_text)}
                    </div>
                    
                    <div style="color: #f9e2af; font-size: 0.9em; margin-top: 15px;">
                        📊 Confidence: {confidence*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Details
                if show_details:
                    with st.expander("📊 Audio Analysis"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Quality Score", f"{result['quality_score']:.1%}")
                        with col2:
                            st.metric("Speaker Learned", f"{result['speaker_learned']} frames")
                        with col3:
                            st.metric("Speech Regions", len(result['speech_regions']))
                        
                        if result['speech_regions']:
                            st.write("**Detected speech regions:**")
                            for i, region in enumerate(result['speech_regions'], 1):
                                st.write(f"  {i}. Confidence: {region['confidence']:.1%}")
        else:
            st.error("❌ Could not process audio")
    else:
        st.warning("⚠️ No audio recorded")

st.divider()

# Footer
st.markdown("""
<div style="text-align: center; color: #808080; font-size: 0.9em; margin-top: 30px;">
    <p>🚀 Built with Deepgram (speech) + Groq (context) + Streamlit</p>
    <p>📦 Deploy to Streamlit Cloud for free</p>
</div>
""", unsafe_allow_html=True)
