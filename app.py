import os
import streamlit as st
import tempfile
import subprocess
import numpy as np
import wave
from io import BytesIO

# --- Configuration & Setup ---
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

# --- FFmpeg Utility (Essential for Audio Extraction) ---

def extract_audio_ffmpeg(video_path, output_wav_path):
    """
    Extracts the audio stream from a video file into an uncompressed WAV format (pcm_s16le).
    This format is required for accurate DSSS decoding.
    """
    try:
        # -y: overwrite; -vn: no video; -acodec pcm_s16le: uncompressed 16-bit WAV (REQUIRED MATCH)
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            output_wav_path
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg extraction failed! Check FFmpeg installation and error details: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        st.error("FFmpeg not found. Please ensure it is installed and in your system PATH.")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
        raise

# --- DSSS Utility Functions (CRITICAL: MUST MATCH EMBEDDER) ---

def generate_pn_sequence(duration_samples):
    """
    Generates the exact same Pseudo-Noise (PN) sequence used for spreading.
    This function must be IDENTICAL to the one used in the OTT embedding app.
    The seed ensures the sequence is reproducible.
    """
    # Use the FIXED SEED used in the embedder (42 in the previous code)
    np.random.seed(42) 
    # Generate random +1 or -1 values, based on the total number of samples
    # The spreading code is a sequence of chips (+1 or -1)
    return (np.random.randint(0, 2, duration_samples) * 2 - 1).astype(np.float64)

# --- DSSS Extraction (Decoding) Function ---

@st.cache_data
def extract_watermark_dsss(input_wav):
    """
    Performs DSSS decoding by correlating the audio signal with the secret PN code.
    Returns the extracted User ID string or a detailed error message.
    """
    try:
        with wave.open(input_wav, "rb") as wav:
            frames = wav.readframes(wav.getparams().nframes)

        # Unpack frames into a NumPy array of 16-bit integers
        watermarked_samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)

    except Exception as e:
        return f"Error reading audio file: {e}"

    # 1. Prepare Parameters (Must match embedding parameters)
    # 1 Signature bit + 8 User ID bits = 9 bits
    payload_length = 9 
    total_samples = len(watermarked_samples)
    
    # Calculate spreading factor (samples per embedded bit)
    spreading_factor = int(np.floor(total_samples / payload_length))
    
    # Define a threshold for detection (tune this based on observed noise/compression)
    detection_threshold = 15.0 
    
    if spreading_factor < 100:
        return "Extraction Failed: Audio is too short for required DSSS processing gain."
    
    # 2. Generate the Secret Spreading Code (PN Sequence)
    pn_sequence = generate_pn_sequence(total_samples)
    
    # 3. Correlation and De-Spreading
    extracted_bits = []
    correlation_values = []

    for i in range(payload_length):
        start_index = i * spreading_factor
        end_index = (i + 1) * spreading_factor
        
        # Get the segment of the watermarked signal and PN sequence
        signal_segment = watermarked_samples[start_index:end_index]
        pn_segment = pn_sequence[start_index:end_index]
        
        # Correlation: Mean of Element-wise Multiplication (De-spreading)
        correlation_value = np.mean(signal_segment * pn_segment)
        correlation_values.append(correlation_value)
        
        # Determine the extracted data bit based on the sign
        extracted_bit = 1 if correlation_value > 0 else 0
        extracted_bits.append(extracted_bit)

    # 4. Decode the Extracted Bits and Validate
    
    # Check the Signature Bit and ensure correlation is above threshold
    signature_correlation = correlation_values[0]
    
    if abs(signature_correlation) < detection_threshold:
        return f"Watermark not detected: Correlation magnitude ({abs(signature_correlation):.2f}) is below threshold ({detection_threshold:.2f})."

    # The first bit is the signature bit (should be +1, correlating to a positive value)
    user_id_bits = extracted_bits[1:]

    # Convert the 8 User ID bits back to a number
    try:
        binary_string = "".join(map(str, user_id_bits))
        extracted_user_id = int(binary_string, 2)
        
        # Successful Extraction: Return the ID and the correlation strength
        return f"{extracted_user_id}|{abs(signature_correlation):.2f}"
    except Exception:
        return "Watermark detected, but decoding failed (Corrupted ID)."


# --- Streamlit Detector App ---

def main():
    st.set_page_config(page_title="DSSS Watermark Detector", layout="wide")
    st.title("🛡️ DSSS Watermark Detector")
    st.markdown("Upload a video previously watermarked by the DSSS process to extract the hidden User ID.")
    st.warning("Ensure the DSSS parameters (PN Code seed, spreading factor) match the embedder's code exactly.")
    
    uploaded_file = st.file_uploader("Choose a Watermarked Video File", type=list(ALLOWED_EXTENSIONS))

    if uploaded_file:
        video_name = uploaded_file.name
        video_bytes = uploaded_file.read()
        
        st.subheader("Video Preview")
        st.video(video_bytes, format='video/mp4')
        
        if st.button(f"Analyze '{video_name}' for Watermark"):
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Save uploaded video to a temporary path
                temp_video_path = os.path.join(temp_dir, video_name)
                with open(temp_video_path, "wb") as f:
                    f.write(video_bytes)
                
                # 2. Define path for temporary audio
                temp_audio_wav = os.path.join(temp_dir, "extracted_audio.wav")
                
                st.info("Starting audio extraction...")
                
                try:
                    # 3. Extract audio from video
                    extract_audio_ffmpeg(temp_video_path, temp_audio_wav)
                    st.success("Audio extracted successfully (PCM 16-bit WAV).")
                    
                    # 4. Extract Watermark using DSSS decoder
                    with st.spinner("Decoding DSSS watermark via correlation..."):
                        result = extract_watermark_dsss(temp_audio_wav)
                    
                    st.header("Extraction Results")
                    
                    if "|" in result:
                        # Success format: "ID|CorrelationValue"
                        extracted_id, correlation = result.split('|')
                        st.balloons()
                        st.success(f"✅ **Watermark Successfully Detected and Decoded!**")
                        st.markdown(f"### Extracted User ID: `{extracted_id}`")
                        st.markdown(f"**Correlation Strength (Metric of Confidence):** `{correlation}`")
                        st.markdown("This indicates the video was downloaded by the user with the ID above.")
                    else:
                        # Failure format: Error message
                        st.error("❌ **Watermark Extraction Failed.**")
                        st.markdown(f"**Detailed Message:** `{result}`")
                        st.markdown("This video may not contain the expected watermark or the audio has been too heavily compressed.")

                except Exception as e:
                    # This catches errors raised during FFmpeg and passes them up
                    st.error(f"FATAL ERROR during processing. See console for details.")

if __name__ == "__main__":
    main()
