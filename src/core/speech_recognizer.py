from faster_whisper import WhisperModel
import os
import time

class SpeechRecognizer:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        # Lazy loading or explicit loading can be done. 
        # We'll load immediately to fail fast if model download fails, 
        # but in GUI it might freeze. Better to load on first use or in background.
        # For simplicity, we assume the user accepts a startup delay.
        print(f"Loading Whisper model: {model_size} on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
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
