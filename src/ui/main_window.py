import sys
import numpy as np
import time
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton, QSlider, QCheckBox, 
                             QTextEdit, QGroupBox, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

from src.core.config_manager import ConfigManager
from src.core.audio_capture import AudioCapture
from src.core.speech_recognizer import SpeechRecognizer
from src.core.translator import Translator
from src.ui.subtitle_window import SubtitleWindow

class TranslationWorker(QThread):
    text_updated = pyqtSignal(str, str) # original, translated
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, config_manager):
        super().__init__()
        self.cm = config_manager
        self.is_running = False
        
        # Core modules
        self.audio_capture = AudioCapture()
        self.recognizer = None
        self.translator = None
        
        # VAD parameters
        self.silence_threshold = 0.01
        self.min_duration = 2.0 # seconds
        self.max_duration = 10.0 # seconds
        self.silence_duration = 0.5 # seconds of silence to trigger

    def run(self):
        self.is_running = True
        
        try:
            # Initialize modules
            self.status_updated.emit("Initializing Speech Recognizer...")
            rec_config = self.cm.get("recognition", {})
            self.recognizer = SpeechRecognizer(
                model_size=rec_config.get("model_size", "base"),
                device=rec_config.get("device", "cpu")
            )
            
            self.status_updated.emit("Initializing Translator...")
            trans_config = self.cm.get("translation", {})
            self.translator = Translator(
                source=trans_config.get("source_lang", "auto"),
                target=trans_config.get("target_lang", "zh-CN")
            )
            
            # Configure Audio
            audio_config = self.cm.get("audio", {})
            self.audio_capture.device_index = audio_config.get("device_index")
            self.audio_capture.start()
            
            self.status_updated.emit("Listening...")
            
            audio_buffer = []
            current_duration = 0.0
            silence_counter = 0.0
            
            while self.is_running:
                chunk = next(self.audio_capture.get_data()) # Blocking
                if chunk is None: continue
                
                audio_buffer.append(chunk)
                chunk_duration = len(chunk) / self.audio_capture.sample_rate
                current_duration += chunk_duration
                
                # Simple VAD (RMS)
                rms = np.sqrt(np.mean(chunk**2))
                if rms < self.silence_threshold:
                    silence_counter += chunk_duration
                else:
                    silence_counter = 0.0
                
                # Trigger conditions
                should_transcribe = False
                if current_duration > self.min_duration and silence_counter > self.silence_duration:
                    should_transcribe = True
                elif current_duration > self.max_duration:
                    should_transcribe = True
                    
                if should_transcribe:
                    # Concatenate and transcribe
                    full_audio = np.concatenate(audio_buffer, axis=0)
                    full_audio = full_audio.flatten() # Ensure 1D
                    
                    self.status_updated.emit("Transcribing...")
                    text, _ = self.recognizer.transcribe(full_audio)
                    
                    if text and text.strip():
                        self.status_updated.emit(f"Recognized: {text}")
                        trans_text = self.translator.translate(text)
                        self.text_updated.emit(text, trans_text)
                    
                    # Reset
                    audio_buffer = []
                    current_duration = 0.0
                    silence_counter = 0.0
                    self.status_updated.emit("Listening...")
                    
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.audio_capture.stop()
            self.status_updated.emit("Stopped.")

    def stop(self):
        self.is_running = False
        self.audio_capture.stop()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trae Real-time Translation")
        self.resize(400, 600)
        
        self.cm = ConfigManager()
        self.subtitle_window = SubtitleWindow(self.cm.get("display", {}))
        self.worker = None
        
        self.init_ui()
        self.subtitle_window.show()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # --- Control Group ---
        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout()
        
        self.btn_start = QPushButton("Start Translation")
        self.btn_start.clicked.connect(self.toggle_translation)
        self.btn_start.setStyleSheet("background-color: #38A169; color: white; font-weight: bold; padding: 10px;")
        
        control_layout.addWidget(self.btn_start)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # --- Audio Settings ---
        audio_group = QGroupBox("Audio Source")
        audio_layout = QVBoxLayout()
        
        self.combo_devices = QComboBox()
        self.refresh_devices()
        self.combo_devices.currentIndexChanged.connect(self.save_audio_settings)
        
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)
        
        audio_layout.addWidget(QLabel("Select Loopback/Input Device:"))
        audio_layout.addWidget(self.combo_devices)
        audio_layout.addWidget(refresh_btn)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        # --- Language Settings ---
        lang_group = QGroupBox("Language")
        lang_layout = QHBoxLayout()
        
        self.combo_src = QComboBox()
        self.combo_src.addItems(["auto", "en", "zh", "ja", "ko"])
        self.combo_src.setCurrentText(self.cm.get("translation", {}).get("source_lang", "auto"))
        
        self.combo_target = QComboBox()
        self.combo_target.addItems(["zh-CN", "en", "ja", "ko"])
        self.combo_target.setCurrentText(self.cm.get("translation", {}).get("target_lang", "zh-CN"))
        
        # Connect save
        self.combo_src.currentTextChanged.connect(self.save_lang_settings)
        self.combo_target.currentTextChanged.connect(self.save_lang_settings)

        lang_layout.addWidget(QLabel("Source:"))
        lang_layout.addWidget(self.combo_src)
        lang_layout.addWidget(QLabel("Target:"))
        lang_layout.addWidget(self.combo_target)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)
        
        # --- Appearance ---
        ui_group = QGroupBox("Appearance")
        ui_layout = QVBoxLayout()
        
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(int(self.cm.get("display", {}).get("opacity", 0.7) * 100))
        self.slider_opacity.valueChanged.connect(self.update_appearance)
        
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(12, 72)
        self.spin_font_size.setValue(self.cm.get("display", {}).get("font_size", 24))
        self.spin_font_size.valueChanged.connect(self.update_appearance)
        
        ui_layout.addWidget(QLabel("Opacity:"))
        ui_layout.addWidget(self.slider_opacity)
        ui_layout.addWidget(QLabel("Font Size:"))
        ui_layout.addWidget(self.spin_font_size)
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)
        
        # --- Logs ---
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_area)

    def refresh_devices(self):
        self.combo_devices.clear()
        capture = AudioCapture()
        devices = capture.list_devices()
        
        default_index = self.cm.get("audio", {}).get("device_index")
        current_index = 0
        
        for i, dev in enumerate(devices):
            name = f"[{dev['index']}] {dev['name']} ({dev['hostapi']})"
            self.combo_devices.addItem(name, dev['index'])
            if default_index is not None and dev['index'] == default_index:
                current_index = i
        
        self.combo_devices.setCurrentIndex(current_index)
        
    def save_audio_settings(self):
        idx = self.combo_devices.currentData()
        if idx is not None:
            self.cm.set("audio", "device_index", idx)

    def save_lang_settings(self):
        self.cm.set("translation", "source_lang", self.combo_src.currentText())
        self.cm.set("translation", "target_lang", self.combo_target.currentText())
        
    def update_appearance(self):
        opacity = self.slider_opacity.value() / 100.0
        font_size = self.spin_font_size.value()
        
        display_config = self.cm.get("display", {})
        display_config["opacity"] = opacity
        display_config["font_size"] = font_size
        self.cm.set("display", "opacity", opacity)
        self.cm.set("display", "font_size", font_size)
        
        # Update live
        self.subtitle_window.config = display_config
        central = self.subtitle_window.centralWidget()
        central.setStyleSheet(f"background-color: rgba(0, 0, 0, {int(opacity * 255)}); border-radius: 10px;")
        self.subtitle_window.update_styles()

    def toggle_translation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.btn_start.setText("Start Translation")
            self.btn_start.setStyleSheet("background-color: #38A169; color: white; font-weight: bold; padding: 10px;")
            self.log("Translation stopped.")
        else:
            self.worker = TranslationWorker(self.cm)
            self.worker.text_updated.connect(self.subtitle_window.update_text)
            self.worker.status_updated.connect(self.log)
            self.worker.error_occurred.connect(lambda e: self.log(f"Error: {e}"))
            self.worker.start()
            self.btn_start.setText("Stop Translation")
            self.btn_start.setStyleSheet("background-color: #E53E3E; color: white; font-weight: bold; padding: 10px;")
            self.log("Starting translation...")

    def log(self, message):
        self.log_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # Auto scroll
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.subtitle_window.close()
        event.accept()
