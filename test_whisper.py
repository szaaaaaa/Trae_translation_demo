import sys
import os

print("Testing faster_whisper import...")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")

try:
    print("\n1. Importing ctranslate2...")
    import ctranslate2
    print(f"   ctranslate2 version: {ctranslate2.__version__}")

    print("\n2. Importing faster_whisper...")
    from faster_whisper import WhisperModel
    print("   faster_whisper imported successfully!")

    print("\n3. Creating WhisperModel (this will download model if not cached)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("   Model created successfully!")

    print("\n=== ALL TESTS PASSED ===")

except Exception as e:
    import traceback
    print(f"\nERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

input("\nPress Enter to exit...")
