import io
import os
import re
import time
import numpy as np
import scipy.io.wavfile as wav
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import webrtcvad
import noisereduce as nr
from unidecode import unidecode

# ============================================================
# 🔑 API KEY (from Streamlit Secrets)
# ============================================================
DEEPGRAM_API_KEY = None
try:
    DEEPGRAM_API_KEY = st.secrets["DEEPGRAM_API_KEY"]
except Exception:
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    st.error("❌ DEEPGRAM_API_KEY not found. Set it in Streamlit Secrets.")
    st.stop()

try:
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
except Exception as e:
    st.error(f"❌ Deepgram init failed: {e}")
    st.stop()

# ============================================================
# 🧠 HUMAN-LIKE AUDIO PROCESSING
# ============================================================
def preprocess_audio_like_human(raw_bytes):
    try:
        sample_rate, audio = wav.read(io.BytesIO(raw_bytes))
        if sample_rate != 16000:
            from scipy import signal
            audio_resampled = signal.resample(audio, int(len(audio) * 16000 / sample_rate))
            audio = audio_resampled.astype(np.int16)
            sample_rate = 16000

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        audio_bytes = audio.tobytes()
        vad = webrtcvad.Vad(2)
        frame_duration_ms = 20
        frame_bytes = int(sample_rate * frame_duration_ms / 1000) * 2

        speech_frames = []
        for i in range(0, len(audio_bytes) - frame_bytes, frame_bytes):
            frame = audio_bytes[i:i + frame_bytes]
            if vad.is_speech(frame, sample_rate):
                speech_frames.append(frame)

        if not speech_frames:
            return None

        speech_audio = b''.join(speech_frames)
        speech_array = np.frombuffer(speech_audio, dtype=np.int16)

        audio_float = speech_array.astype(np.float32) / 32768.0
        cleaned = nr.reduce_noise(
            y=audio_float,
            sr=sample_rate,
            prop_decrease=0.6,
            n_fft=512,
        )
        cleaned_int16 = np.clip(cleaned * 32768, -32768, 32767).astype(np.int16)

        output = io.BytesIO()
        wav.write(output, sample_rate, cleaned_int16)
        return output.getvalue()

    except Exception as e:
        st.warning(f"Audio processing error: {e}")
        return None

# ============================================================
# 🎙️ DEEPGRAM TRANSCRIPTION
# ============================================================
def transcribe_human_like(processed_audio_bytes):
    try:
        payload = FileSource(buffer=processed_audio_bytes, mimetype='audio/wav')
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            punctuate=True,
            utterances=True,
            diarize=True,
            language="multi",
        )
        response = client.listen.rest.v('1').transcribe_file(payload, options)

        if response and hasattr(response, 'results'):
            channel = response.results.channels[0]
            alternatives = channel.alternatives
            if alternatives:
                alt = alternatives[0]
                transcript = alt.transcript
                confidence = alt.confidence

                if re.search(r'[\u0600-\u06FF]', transcript):
                    transcript = unidecode(transcript)

                return transcript.strip(), confidence
        return "", 0.0
    except Exception as e:
        st.error(f"Deepgram error: {e}")
        return "", 0.0

# ============================================================
# 📱 STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Human-Like STT", layout="centered")
st.title("🧠 Human‑Like Speech‑to‑Text Agent")
st.caption("Hears like a human – ignores noise, focuses on speaker")

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

audio = mic_recorder(
    start_prompt="🎤 Start Speaking",
    stop_prompt="⏹️ Stop",
    just_once=True,
    format="wav",
    key="mic"
)

if audio and 'bytes' in audio:
    raw_bytes = audio['bytes']
    if raw_bytes:
        with st.spinner("⏳ Listening like a human..."):
            cleaned = preprocess_audio_like_human(raw_bytes)
            if cleaned is None:
                st.warning("⚠️ No speech detected. Please speak clearly.")
            else:
                text, confidence = transcribe_human_like(cleaned)
                if text:
                    st.session_state.last_text = text
                    st.success(f"✅ Done (Confidence: {confidence:.2f})")
                else:
                    st.warning("⚠️ Could not recognize speech. Try again.")

st.divider()
st.subheader("📝 Transcription")
if st.session_state.last_text:
    st.code(st.session_state.last_text, language="text")
else:
    st.info("Your transcription will appear here.")

if st.button("🗑️ Clear"):
    st.session_state.last_text = ""
    st.rerun()
