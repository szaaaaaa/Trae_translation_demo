import sys
import os
import traceback

# Set OpenMP environment variables BEFORE any ctranslate2/faster_whisper imports
# This helps avoid DLL conflicts in packaged applications
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Single thread to avoid OpenMP DLL conflicts

def test_dependencies():
    """Test critical dependencies before starting GUI"""
    print("Testing dependencies...")

    try:
        print("  1. Testing ctranslate2...")
        import ctranslate2
        print(f"     ctranslate2 OK (version: {ctranslate2.__version__})")

        print("  2. Testing faster_whisper...")
        from faster_whisper import WhisperModel
        print("     faster_whisper OK")

        print("Dependencies OK!\n")
        return True
    except Exception as e:
        print(f"\nDependency error: {e}")
        traceback.print_exc()
        return False

def main():
    from PyQt6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)

    # Set app style/font if needed
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        print("Starting Trae Translation App...")
        print(f"Python: {sys.version}")
        print(f"Executable: {sys.executable}")
        print()

        if not test_dependencies():
            input("Press Enter to exit...")
            sys.exit(1)

        main()
    except Exception as e:
        print("\n" + "="*50)
        print("FATAL ERROR:")
        print("="*50)
        traceback.print_exc()
        print("="*50)
        input("Press Enter to exit...")
        sys.exit(1)
