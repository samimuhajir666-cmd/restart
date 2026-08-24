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
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_mic_recorder import mic_recorder
from collections import deque

# ============================
# 🔧 IMPORT LIBROSA (Critical Fix)
# ============================
try:
    import librosa
except ImportError:
    st.error("❌ librosa not installed. Please run: pip install librosa")
    st.stop()

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

# API Keys — with safe fallback
def get_api_key(var_name):
    try:
        # First try .env
        key = os.getenv(var_name)
        if key:
            return key
        # Then try Streamlit secrets
        return st.secrets.get(var_name, None)
    except Exception:
        return os.getenv(var_name)

DEEPGRAM_API_KEY = get_api_key("DEEPGRAM_API_KEY")
GROQ_API_KEY = get_api_key("GROQ_API_KEY")

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY not found in .env or Secrets")
    st.stop()

# ============================
# 👂 VOICE ACTIVITY DETECTOR
# ============================

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 20
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        self.speech_floor_db = 35  # Minimum speech level
        self.noise_ceiling_db = 25  # Maximum noise level
        self.vad_history = deque(maxlen=30)
        
        self.noise_spectrum = None
        self.speech_spectrum = None
        self.learning_frames = 0

    def compute_mfcc_features(self, audio_chunk):
        try:
            # Convert int16 to float32 normalized between -1.0 and 1.0 for librosa
            y_float = audio_chunk.astype(np.float32) / 32768.0

            mel_spec = librosa.feature.melspectrogram(
                y=y_float,
                sr=self.sample_rate,
                n_mels=13
            )
            
            mfcc = librosa.feature.mfcc(
                y=y_float,
                sr=self.sample_rate,
                n_mfcc=13
            )
            
            return {
                'mel_energy': float(np.mean(mel_spec)),
                'mel_variance': float(np.std(mel_spec)),
                'mfcc_delta': float(np.mean(np.abs(np.diff(mfcc, axis=1)))) if mfcc.shape[1] > 1 else 0.0
            }
        except Exception:
            return self._compute_fallback_features(audio_chunk)

    def _compute_fallback_features(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        energy_db = 20 * np.log10(rms / 32768.0 + 1e-10)
        
        zcr = np.sum(np.abs(np.diff(np.sign(audio_chunk)))) / (2 * len(audio_chunk) + 1e-10)
        
        spectrum = np.abs(np.fft.fft(audio_chunk))
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
        entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10))
        
        return {
            'energy': energy_db,
            'zcr': zcr,
            'entropy': entropy
        }

    def is_speech(self, audio_chunk):
        features = self.compute_mfcc_features(audio_chunk)
        factors = []
        
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        energy_db = 20 * np.log10(rms / 32768.0 + 1e-10)
        factors.append(energy_db > self.speech_floor_db)
        
        if 'mel_variance' in features:
            factors.append(features['mel_variance'] > 0.01)
        
        if 'mfcc_delta' in features:
            factors.append(features['mfcc_delta'] > 0.1)
        
        speech_score = sum(factors) / len(factors) if factors else 0.0
        self.vad_history.append(speech_score > 0.5)
        
        if len(self.vad_history) > 5:
            return (sum(self.vad_history) / len(self.vad_history)) > 0.4
        
        return speech_score > 0.5


# ============================
# 🗣️ SPEAKER PRIORITIZER
# ============================

class SpeakerPrioritizer:
    def __init__(self):
        self.primary_speaker_profile = None
        self.learning_count = 0
        self.max_learning = 30
        self.confidence_threshold = 0.6

    def extract_pitch_features(self, audio_chunk):
        try:
            y_float = audio_chunk.astype(np.float64) / 32768.0
            f0 = librosa.yin(
                y_float,
                fmin=50,
                fmax=400,
                sr=16000
            )
            
            voiced = f0[~np.isnan(f0)]
            if len(voiced) > 0:
                return {
                    'pitch_mean': float(np.mean(voiced)),
                    'pitch_std': float(np.std(voiced)),
                    'pitch_range': (float(np.min(voiced)), float(np.max(voiced)))
                }
        except Exception:
            pass
        return None

    def learn_speaker(self, audio_chunk):
        if self.learning_count >= self.max_learning:
            return
        
        pitch_feat = self.extract_pitch_features(audio_chunk)
        if pitch_feat:
            if self.primary_speaker_profile is None:
                self.primary_speaker_profile = pitch_feat
            else:
                alpha = 0.1
                old = self.primary_speaker_profile
                self.primary_speaker_profile = {
                    'pitch_mean': alpha * pitch_feat['pitch_mean'] + (1 - alpha) * old['pitch_mean'],
                    'pitch_std': alpha * pitch_feat['pitch_std'] + (1 - alpha) * old['pitch_std'],
                    'pitch_range': pitch_feat['pitch_range']
                }
        self.learning_count += 1

    def is_primary_speaker(self, audio_chunk):
        if self.learning_count < 5:
            return True
        
        pitch_feat = self.extract_pitch_features(audio_chunk)
        if not pitch_feat or not self.primary_speaker_profile:
            return True
        
        pitch_diff = abs(pitch_feat['pitch_mean'] - self.primary_speaker_profile['pitch_mean'])
        tolerance = self.primary_speaker_profile['pitch_std'] * 2 + 1e-5
        
        return pitch_diff < tolerance


# ============================
# 🔊 NOISE SUPPRESSION ENGINE
# ============================

class AdaptiveNoiseGate:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_floor = 25
        self.noise_floor_history = deque(maxlen=50)
        self.adaptation_rate = 0.05

    def update_noise_floor(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        chunk_db = 20 * np.log10(rms / 32768.0 + 1e-10)
        
        self.noise_floor_history.append(chunk_db)
        if len(self.noise_floor_history) > 20:
            new_floor = np.percentile(list(self.noise_floor_history), 25)
            self.noise_floor = self.adaptation_rate * new_floor + (1 - self.adaptation_rate) * self.noise_floor

    def suppress(self, audio_chunk):
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2) + 1e-10)
        chunk_db = 20 * np.log10(rms / 32768.0 + 1e-10)
        
        if chunk_db < self.noise_floor + 5:
            attenuation = max(0.0, (self.noise_floor + 10.0 - chunk_db) / 10.0)
            return (audio_chunk.astype(np.float64) * (1.0 - attenuation * 0.7)).astype(np.int16)
        
        return audio_chunk


# ============================
# 📝 TRANSCRIPTION QUALITY TRACKER
# ============================

class TranscriptionQualityTracker:
    def __init__(self):
        self.segment_confidences = []
        self.segment_texts = []
        self.unclear_threshold = 0.4

    def add_segment(self, text, confidence):
        self.segment_texts.append(text)
        self.segment_confidences.append(confidence)

    def mark_unclear(self, transcription, confidences):
        if not confidences:
            return transcription
        
        avg_conf = float(np.mean(confidences))
        if avg_conf < self.unclear_threshold:
            return f"[inaudible: {transcription}]" if transcription else "[inaudible]"
        if avg_conf < 0.6:
            return f"{transcription} [low confidence]"
        
        return transcription


# ============================
# 🎙️ DEEPGRAM TRANSCRIPTION
# ============================

def transcribe_with_deepgram(audio_bytes, language="ur"):
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
    if not GROQ_API_KEY or confidence > 0.8:
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
    except Exception:
        return transcription


# ============================
# 📊 AUDIO QUALITY ASSESSMENT
# ============================

def assess_audio_quality(audio_data, sample_rate):
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2) + 1e-10)
    energy_db = 20 * np.log10(rms / 32768.0 + 1e-10)
    
    chunk_len = max(1, sample_rate // 10)
    step_len = max(1, sample_rate // 20)
    
    sorted_energies = sorted([
        20 * np.log10(np.sqrt(np.mean(audio_data[i:i+chunk_len].astype(np.float64)**2) + 1e-10) / 32768.0 + 1e-10)
        for i in range(0, max(1, len(audio_data) - chunk_len), step_len)
    ])
    
    noise_floor = np.mean(sorted_energies[:max(1, len(sorted_energies)//4)]) if sorted_energies else -60
    snr = energy_db - noise_floor
    
    quality = 0.0
    if 40 < energy_db < 60:
        quality += 0.3
    elif 35 < energy_db < 65:
        quality += 0.2
        
    if snr > 10:
        quality += 0.4
    elif snr > 5:
        quality += 0.2
        
    if np.max(np.abs(audio_data)) < 32000:
        quality += 0.3
        
    return min(1.0, float(quality))


# ============================
# 🎙️ MAIN PROCESSING PIPELINE
# ============================

def process_audio_human_like(audio_bytes, use_groq_context=True, show_details=False):
    try:
        audio_file = io.BytesIO(audio_bytes)
        sample_rate, audio_data = wav.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_data = audio_data.astype(np.int16)
        quality_score = assess_audio_quality(audio_data, sample_rate)
        
        vad = VoiceActivityDetector(sample_rate)
        speaker = SpeakerPrioritizer()
        noise_gate = AdaptiveNoiseGate(sample_rate)
        
        frame_size = max(1, sample_rate // 50)
        speech_regions = []
        processed_frames = []
        
        current_region_start = None
        current_region_confidence = []
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i+frame_size]
            is_speech = vad.is_speech(frame)
            noise_gate.update_noise_floor(frame)
            
            if is_speech:
                speaker.learn_speaker(frame)
                is_primary = speaker.is_primary_speaker(frame)
                
                if is_primary:
                    suppressed = noise_gate.suppress(frame)
                    processed_frames.append(suppressed)
                    
                    if current_region_start is None:
                        current_region_start = i
                    current_region_confidence.append(quality_score)
            else:
                if current_region_start is not None:
                    region_confidence = np.mean(current_region_confidence) if current_region_confidence else 0
                    speech_regions.append({
                        'start': current_region_start,
                        'end': i,
                        'confidence': region_confidence
                    })
                    current_region_start = None
                    current_region_confidence = []
                
                processed_frames.append(frame)
        
        if not processed_frames:
            return None
        
        processed_audio = np.concatenate(processed_frames)
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
    'السلام علیکم': 'Assalam alaikum',
    'السلام': 'Assalam',
    'آپ کیسے ہیں': 'Aap kaise hain',
    'خدا حافظ': 'Khuda hafiz',
    'ہیلو': 'Hello',
    'شکریہ': 'Shukriya',
    'نمسٹے': 'Namaste',
    'جی': 'Haan',
    'نہیں': 'Nahi',
}

def transliterate(text):
    for urdu, roman in URDU_TO_ROMAN.items():
        text = text.replace(urdu, roman)
    text = re.sub(r'[\u0600-\u06FF]', '', text)
    return text


# ============================
# 🖥️ STREAMLIT INTERFACE
# ============================

st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1>👂 Human-Like Speech-to-Text Agent</h1>
    <p style="font-size: 1.1em; color: #808080;">
    Listens naturally • Ignores background noise • Focuses on main speaker
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("🧠 How This Works"):
    st.markdown("""
    **Like human ears, this agent:**
    
    1. **Listens continuously** - Monitors all audio
    2. **Detects speech** - Identifies human speech vs background noise
    3. **Learns your voice** - Focuses on primary speaker
    4. **Filters noise** - Suppresses AC, fans, background chatter
    5. **Understands context** - Uses nearby words to clarify unclear parts
    6. **Marks unclear sections** - Says [inaudible] instead of guessing
    """)

col1, col2, col3 = st.columns(3)

with col1:
    use_context = st.checkbox("🧠 Context-Aware (Groq)", value=True)

with col2:
    show_details = st.checkbox("📊 Show Details", value=False)

with col3:
    lang = st.selectbox("🌐 Language", ["Urdu (اردو)", "Hindi (ہندی)", "English"], index=0)

lang_code = {"Urdu (اردو)": "ur", "Hindi (ہندی)": "hi", "English": "en"}[lang]

st.divider()

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

if audio_output and audio_output.get("bytes"):
    audio_bytes = audio_output.get("bytes")
    
    with st.spinner("👂 Listening like human ears..."):
        result = process_audio_human_like(
            audio_bytes,
            use_groq_context=use_context,
            show_details=show_details
        )
    
    if result:
        with st.spinner("📝 Transcribing..."):
            trans = transcribe_with_deepgram(result['processed_bytes'], language=lang_code)
        
        if trans:
            text = trans['text']
            confidence = trans['confidence']
            
            if use_context and GROQ_API_KEY:
                with st.spinner("🧠 Applying context..."):
                    text = enhance_with_groq_context(
                        text,
                        confidence,
                        result['quality_score']
                    )
            
            roman_text = transliterate(text) if lang_code != "en" else text
            
            if confidence < 0.5:
                roman_text = f"{roman_text}\n\n⚠️ [Note: Audio was unclear, confidence {confidence*100:.0f}%]"
            
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
            
            if show_details:
                with st.expander("📊 Audio Analysis"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Quality Score", f"{result['quality_score']:.1%}")
                    with c2:
                        st.metric("Speaker Learned", f"{result['speaker_learned']} frames")
                    with c3:
                        st.metric("Speech Regions", len(result['speech_regions']))
    else:
        st.error("❌ Could not process audio")

st.divider()

st.markdown("""
<div style="text-align: center; color: #808080; font-size: 0.9em; margin-top: 30px;">
    <p>🚀 Built with Deepgram (speech) + Groq (context) + Streamlit</p>
</div>
""", unsafe_allow_html=True)
