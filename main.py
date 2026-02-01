import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
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
    print("Starting Trae Translation App...")
    main()
