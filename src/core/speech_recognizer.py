import os
import sys
import time

# Set OpenMP environment variables BEFORE importing ctranslate2/faster_whisper
# This helps avoid DLL conflicts in packaged applications
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Single thread to avoid OpenMP DLL conflicts

from faster_whisper import WhisperModel

class SpeechRecognizer:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

        print(f"Loading Whisper model: {model_size} on {device}...")
        sys.stdout.flush()  # Ensure output is visible before potential crash

        try:
            print(f"  compute_type: {compute_type}")
            print(f"  Creating WhisperModel...")
            sys.stdout.flush()

            # Use int8 for CPU to avoid potential issues
            if device == "cpu":
                compute_type = "int8"

            # Use single thread to avoid OpenMP DLL conflicts in packaged app
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=1
            )
            print("Whisper model loaded successfully.")
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"Error loading Whisper model: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            raise e

    def transcribe(self, audio_data, language=None):
        """
        Transcribe audio data.
        :param audio_data: numpy array of float32
        :param language: language code (e.g., 'en', 'zh'), or None for auto-detection
        :return: (text, info)
        """
        if self.model is None:
            return "", None

        try:
            # beam_size=5 is standard. vad_filter=True helps with silence.
            segments, info = self.model.transcribe(
                audio_data, 
                beam_size=5, 
                language=language,
                vad_filter=True
            )
            
            # segments is a generator, we must iterate to get result
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text)
            
            full_text = " ".join(text_segments).strip()
            return full_text, info
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", None
