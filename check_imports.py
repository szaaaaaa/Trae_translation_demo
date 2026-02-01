import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Checking imports...")
    from src.core.config_manager import ConfigManager
    print("ConfigManager imported.")
    
    from src.core.audio_capture import AudioCapture
    print("AudioCapture imported.")
    
    # Mocking sounddevice query for CI/CD environments if needed, but here we just import class
    
    from src.core.speech_recognizer import SpeechRecognizer
    print("SpeechRecognizer imported.")
    
    from src.core.translator import Translator
    print("Translator imported.")
    
    from src.ui.main_window import MainWindow
    print("MainWindow imported.")
    
    from src.ui.subtitle_window import SubtitleWindow
    print("SubtitleWindow imported.")
    
    print("All modules imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
