# 实时翻译应用重构方案：基于 Meta SeamlessM4T v2

## 概述

本文档描述如何将现有的实时翻译应用从 `faster_whisper + 第三方翻译API` 架构重构为基于 **Meta SeamlessM4T v2** 的端到端语音翻译方案。

### 为什么重构？

1. **当前问题**：faster_whisper/ctranslate2 在 PyInstaller 打包后存在 OpenMP DLL 冲突，导致程序崩溃
2. **SeamlessM4T 优势**：
   - 端到端语音翻译，无需分离的识别和翻译模块
   - 通过 HuggingFace Transformers 提供标准化 API
   - 支持 100+ 语言
   - 更好的翻译质量（保留语义和上下文）

---

## 技术选型

### 模型选择：SeamlessM4T v2 Large

- **模型**：`facebook/seamless-m4t-v2-large`
- **来源**：HuggingFace Transformers
- **能力**：
  - 语音输入：101 种语言
  - 文本输入/输出：96 种语言
  - 语音输出：35 种语言
- **硬件要求**：
  - **最低**：NVIDIA GPU 8GB 显存（使用 8-bit 量化）
  - **推荐**：NVIDIA GPU 12GB+ 显存（使用 float16）
  - 支持：RTX 5070 Laptop 8GB、RTX 4060 8GB、RTX 3070 8GB 等
- **CUDA 版本要求**：
  - RTX 50 系列 (Blackwell/SM_120)：**CUDA 12.8+, PyTorch 2.6+**
  - RTX 40/30 系列：CUDA 12.1+, PyTorch 2.0+

### 依赖库

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.36.0
sentencepiece
sounddevice
numpy
PyQt6
```

---

## 项目结构（重构后）

```
Trae_translation/
├── main.py                      # 入口文件
├── requirements.txt             # 依赖列表
├── build.py                     # PyInstaller 打包脚本
├── config.json                  # 配置文件
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config_manager.py    # 配置管理（保留）
│   │   ├── audio_capture.py     # 音频采集（重写）
│   │   └── seamless_translator.py  # 新：SeamlessM4T 翻译器
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py       # 主窗口（重写）
│       └── subtitle_window.py   # 字幕窗口（保留）
```

---

## 核心模块实现

### 1. 依赖安装 (requirements.txt)

```txt
# PyTorch with CUDA support
# 注意：RTX 50 系列 (Blackwell/SM_120) 需要 CUDA 12.8+ 和 PyTorch 2.6+
# 请根据显卡选择正确的 CUDA 版本：
#   - RTX 50 系列 (5070/5080/5090): cu128 或更高
#   - RTX 40/30 系列: cu121 或 cu124
--extra-index-url https://download.pytorch.org/whl/cu128
torch>=2.6.0
torchaudio>=2.6.0

# HuggingFace Transformers
transformers>=4.36.0
sentencepiece
accelerate

# 8-bit 量化支持（8GB 显存必需）
bitsandbytes>=0.41.0

# Audio
sounddevice
numpy

# UI
PyQt6

# Packaging (optional)
pyinstaller
```

### 2. 音频采集模块 (src/core/audio_capture.py)

```python
"""
音频采集模块 - 使用 sounddevice 采集系统音频
"""
import sounddevice as sd
import numpy as np
from typing import Generator, Optional, List, Dict
import threading
import queue

class AudioCapture:
    """音频采集类，支持系统回环和麦克风输入"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_duration: float = 0.5):
        """
        初始化音频采集器

        Args:
            sample_rate: 采样率，SeamlessM4T 要求 16000Hz
            channels: 声道数，单声道
            chunk_duration: 每个音频块的时长（秒）
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

        self.device_index: Optional[int] = None
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False

    def list_devices(self) -> List[Dict]:
        """列出所有可用的音频输入设备"""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate'],
                    'hostapi': sd.query_hostapis(dev['hostapi'])['name']
                })
        return devices

    def _audio_callback(self, indata, frames, time, status):
        """音频流回调函数"""
        if status:
            print(f"Audio status: {status}")
        # 转换为 float32 并放入队列
        audio_data = indata.copy().astype(np.float32)
        if self.channels == 1 and audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        self.audio_queue.put(audio_data)

    def start(self):
        """开始音频采集"""
        if self.is_running:
            return

        self.is_running = True
        self.audio_queue = queue.Queue()

        # 获取设备的原生采样率
        if self.device_index is not None:
            device_info = sd.query_devices(self.device_index)
            native_sr = int(device_info['default_samplerate'])
        else:
            native_sr = self.sample_rate

        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=native_sr,
            channels=self.channels,
            dtype=np.float32,
            blocksize=int(native_sr * self.chunk_duration),
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """停止音频采集"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取一个音频块

        Args:
            timeout: 超时时间（秒）

        Returns:
            音频数据 numpy 数组，如果超时返回 None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_audio_generator(self) -> Generator[np.ndarray, None, None]:
        """返回音频数据生成器"""
        while self.is_running:
            chunk = self.get_audio_chunk()
            if chunk is not None:
                yield chunk
```

### 3. SeamlessM4T 翻译器 (src/core/seamless_translator.py)

```python
"""
SeamlessM4T v2 语音翻译模块
端到端语音识别 + 翻译
"""
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple, Dict, Any
from transformers import AutoProcessor, SeamlessM4Tv2Model

# SeamlessM4T 语言代码映射
LANGUAGE_CODES = {
    # 常用语言
    "auto": None,  # 自动检测
    "zh": "cmn",   # 中文（普通话）
    "en": "eng",   # 英语
    "ja": "jpn",   # 日语
    "ko": "kor",   # 韩语
    "es": "spa",   # 西班牙语
    "fr": "fra",   # 法语
    "de": "deu",   # 德语
    "ru": "rus",   # 俄语
    "ar": "arb",   # 阿拉伯语
    "pt": "por",   # 葡萄牙语
    "it": "ita",   # 意大利语
    "vi": "vie",   # 越南语
    "th": "tha",   # 泰语
    "id": "ind",   # 印尼语
    "ms": "zsm",   # 马来语
    "hi": "hin",   # 印地语
    "tr": "tur",   # 土耳其语
    "pl": "pol",   # 波兰语
    "nl": "nld",   # 荷兰语
    "sv": "swe",   # 瑞典语
    "da": "dan",   # 丹麦语
    "fi": "fin",   # 芬兰语
    "no": "nob",   # 挪威语
    "cs": "ces",   # 捷克语
    "el": "ell",   # 希腊语
    "he": "heb",   # 希伯来语
    "uk": "ukr",   # 乌克兰语
    "ro": "ron",   # 罗马尼亚语
    "hu": "hun",   # 匈牙利语
}

# 目标语言代码（用于翻译输出）
TARGET_LANGUAGE_CODES = {
    "zh-CN": "cmn",
    "zh-TW": "cmn",  # SeamlessM4T 不区分简繁
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "ru": "rus",
}


class SeamlessTranslator:
    """
    SeamlessM4T v2 语音翻译器
    支持语音到文本翻译（S2TT）
    """

    def __init__(self,
                 model_name: str = "facebook/seamless-m4t-v2-large",
                 device: str = "cuda",
                 use_8bit: bool = True):
        """
        初始化翻译器

        Args:
            model_name: HuggingFace 模型名称
            device: 运行设备 ("cuda" 或 "cpu")
            use_8bit: 是否使用 8-bit 量化（8GB 显存建议开启）
        """
        self.device = device
        self.use_8bit = use_8bit
        self.model_name = model_name

        self.processor = None
        self.model = None
        self.sample_rate = 16000  # SeamlessM4T 要求 16kHz

        self._loaded = False

    def load_model(self, progress_callback=None):
        """
        加载模型（可能需要下载）

        Args:
            progress_callback: 进度回调函数，接收 (message: str) 参数
        """
        if self._loaded:
            return

        if progress_callback:
            progress_callback("正在加载 SeamlessM4T 处理器...")

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if progress_callback:
            if self.use_8bit:
                progress_callback("正在加载 SeamlessM4T 模型（8-bit 量化模式，约需 6GB 显存）...")
            else:
                progress_callback("正在加载 SeamlessM4T 模型（首次运行需要下载约 10GB）...")

        # 根据显存大小选择加载方式
        if self.use_8bit and self.device == "cuda":
            # 8-bit 量化加载，适合 8GB 显存
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False
            )

            self.model = SeamlessM4Tv2Model.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            # 标准 float16 加载
            self.model = SeamlessM4Tv2Model.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cuda" and not self.use_8bit:
                self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

        if progress_callback:
            # 显示显存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                progress_callback(f"模型加载完成！显存占用: {allocated:.1f} GB")

    def _ensure_sample_rate(self, audio: np.ndarray,
                            original_sr: int) -> np.ndarray:
        """确保音频采样率为 16kHz"""
        if original_sr != self.sample_rate:
            # 使用 torchaudio 重采样
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampled = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=original_sr,
                new_freq=self.sample_rate
            )
            return resampled.squeeze().numpy()
        return audio

    def translate(self,
                  audio: np.ndarray,
                  source_lang: str = "auto",
                  target_lang: str = "zh-CN",
                  original_sr: int = 16000) -> Tuple[str, str]:
        """
        翻译音频

        Args:
            audio: 音频数据 (numpy array, float32, mono)
            source_lang: 源语言代码 (如 "en", "zh", "auto")
            target_lang: 目标语言代码 (如 "zh-CN", "en")
            original_sr: 音频原始采样率

        Returns:
            (original_text, translated_text) 元组
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载，请先调用 load_model()")

        # 重采样到 16kHz
        audio = self._ensure_sample_rate(audio, original_sr)

        # 确保是 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 获取语言代码
        src_lang_code = LANGUAGE_CODES.get(source_lang, source_lang)
        tgt_lang_code = TARGET_LANGUAGE_CODES.get(target_lang, target_lang)

        # 处理音频输入
        audio_inputs = self.processor(
            audios=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        # 移动到设备
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            # 生成翻译文本
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang_code,
                generate_speech=False,  # 只生成文本
                num_beams=5,
                do_sample=False,
            )

            # 解码文本
            translated_text = self.processor.batch_decode(
                output_tokens[0],
                skip_special_tokens=True
            )[0]

            # 获取原文（ASR）- 如果需要的话
            # 注意：SeamlessM4T 是端到端翻译，没有显式的原文输出
            # 如果需要原文，可以设置 tgt_lang 为源语言
            original_text = ""
            if src_lang_code and src_lang_code != tgt_lang_code:
                # 再次生成，目标语言设为源语言以获得转录
                try:
                    asr_tokens = self.model.generate(
                        **audio_inputs,
                        tgt_lang=src_lang_code if src_lang_code else "eng",
                        generate_speech=False,
                        num_beams=5,
                        do_sample=False,
                    )
                    original_text = self.processor.batch_decode(
                        asr_tokens[0],
                        skip_special_tokens=True
                    )[0]
                except:
                    original_text = "[原文获取失败]"

        return original_text, translated_text

    def translate_speech_to_text(self,
                                  audio: np.ndarray,
                                  target_lang: str = "zh-CN",
                                  original_sr: int = 16000) -> str:
        """
        简化版翻译，只返回翻译结果

        Args:
            audio: 音频数据
            target_lang: 目标语言
            original_sr: 原始采样率

        Returns:
            翻译后的文本
        """
        _, translated = self.translate(audio, "auto", target_lang, original_sr)
        return translated

    def transcribe(self,
                   audio: np.ndarray,
                   language: str = "en",
                   original_sr: int = 16000) -> str:
        """
        语音识别（ASR），不翻译

        Args:
            audio: 音频数据
            language: 语言代码
            original_sr: 原始采样率

        Returns:
            识别的文本
        """
        if not self._loaded:
            raise RuntimeError("模型尚未加载")

        audio = self._ensure_sample_rate(audio, original_sr)

        lang_code = LANGUAGE_CODES.get(language, language)

        audio_inputs = self.processor(
            audios=audio.astype(np.float32),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=lang_code,
                generate_speech=False,
                num_beams=5,
            )
            text = self.processor.batch_decode(
                output_tokens[0],
                skip_special_tokens=True
            )[0]

        return text

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._loaded

    def unload(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 4. 主窗口 (src/ui/main_window.py)

```python
"""
主窗口 - PyQt6 实现
"""
import sys
import numpy as np
import time
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QSlider,
                             QTextEdit, QGroupBox, QSpinBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from src.core.config_manager import ConfigManager
from src.core.audio_capture import AudioCapture
from src.core.seamless_translator import SeamlessTranslator, TARGET_LANGUAGE_CODES
from src.ui.subtitle_window import SubtitleWindow


class ModelLoaderThread(QThread):
    """模型加载线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, translator: SeamlessTranslator):
        super().__init__()
        self.translator = translator

    def run(self):
        try:
            self.translator.load_model(progress_callback=self.progress.emit)
            self.finished.emit(True, "模型加载成功")
        except Exception as e:
            self.finished.emit(False, f"模型加载失败: {str(e)}")


class TranslationWorker(QThread):
    """翻译工作线程"""
    text_updated = pyqtSignal(str, str)  # (original, translated)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, translator: SeamlessTranslator,
                 audio_capture: AudioCapture,
                 config_manager: ConfigManager):
        super().__init__()
        self.translator = translator
        self.audio_capture = audio_capture
        self.cm = config_manager
        self.is_running = False

        # VAD 参数
        self.silence_threshold = 0.01
        self.min_duration = 2.0    # 最小音频时长（秒）
        self.max_duration = 10.0   # 最大音频时长（秒）
        self.silence_duration = 0.5  # 触发翻译的静音时长（秒）

    def run(self):
        self.is_running = True

        try:
            # 获取配置
            trans_config = self.cm.get("translation")
            source_lang = trans_config.get("source_lang", "auto")
            target_lang = trans_config.get("target_lang", "zh-CN")

            audio_config = self.cm.get("audio")
            self.audio_capture.device_index = audio_config.get("device_index")

            # 开始音频采集
            self.audio_capture.start()
            self.status_updated.emit("正在监听...")

            audio_buffer = []
            current_duration = 0.0
            silence_counter = 0.0
            sample_rate = self.audio_capture.sample_rate

            while self.is_running:
                # 获取音频块
                chunk = self.audio_capture.get_audio_chunk(timeout=0.5)
                if chunk is None:
                    continue

                audio_buffer.append(chunk)
                chunk_duration = len(chunk) / sample_rate
                current_duration += chunk_duration

                # 简单 VAD（基于 RMS）
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < self.silence_threshold:
                    silence_counter += chunk_duration
                else:
                    silence_counter = 0.0

                # 判断是否应该翻译
                should_translate = False
                if current_duration > self.min_duration and silence_counter > self.silence_duration:
                    should_translate = True
                elif current_duration > self.max_duration:
                    should_translate = True

                if should_translate and audio_buffer:
                    # 合并音频
                    full_audio = np.concatenate(audio_buffer, axis=0)
                    full_audio = full_audio.flatten()

                    self.status_updated.emit("正在翻译...")

                    try:
                        # 执行翻译
                        original, translated = self.translator.translate(
                            audio=full_audio,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            original_sr=sample_rate
                        )

                        if translated.strip():
                            self.text_updated.emit(original, translated)

                    except Exception as e:
                        self.status_updated.emit(f"翻译错误: {str(e)}")

                    # 重置缓冲区
                    audio_buffer = []
                    current_duration = 0.0
                    silence_counter = 0.0
                    self.status_updated.emit("正在监听...")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
        finally:
            self.audio_capture.stop()
            self.status_updated.emit("已停止")

    def stop(self):
        self.is_running = False
        self.audio_capture.stop()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seamless 实时翻译")
        self.resize(450, 650)

        # 初始化组件
        self.cm = ConfigManager()
        self.audio_capture = AudioCapture()
        # 使用 8-bit 量化，支持 8GB 显存显卡（如 RTX 5070 Laptop）
        self.translator = SeamlessTranslator(device="cuda", use_8bit=True)

        self.subtitle_window = SubtitleWindow(self.cm.config.get("display", {}))
        self.worker = None
        self.model_loader = None

        self.init_ui()
        self.subtitle_window.show()

        # 启动时加载模型
        self.load_model()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- 状态显示 ---
        status_group = QGroupBox("模型状态")
        status_layout = QVBoxLayout()

        self.lbl_model_status = QLabel("模型未加载")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.hide()

        status_layout.addWidget(self.lbl_model_status)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # --- 控制按钮 ---
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()

        self.btn_start = QPushButton("开始翻译")
        self.btn_start.clicked.connect(self.toggle_translation)
        self.btn_start.setEnabled(False)  # 等待模型加载
        self.btn_start.setStyleSheet(
            "background-color: #38A169; color: white; "
            "font-weight: bold; padding: 10px;"
        )

        control_layout.addWidget(self.btn_start)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # --- 音频设置 ---
        audio_group = QGroupBox("音频源")
        audio_layout = QVBoxLayout()

        self.combo_devices = QComboBox()
        self.refresh_devices()
        self.combo_devices.currentIndexChanged.connect(self.save_audio_settings)

        refresh_btn = QPushButton("刷新设备")
        refresh_btn.clicked.connect(self.refresh_devices)

        audio_layout.addWidget(QLabel("选择输入设备:"))
        audio_layout.addWidget(self.combo_devices)
        audio_layout.addWidget(refresh_btn)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        # --- 语言设置 ---
        lang_group = QGroupBox("语言")
        lang_layout = QHBoxLayout()

        self.combo_src = QComboBox()
        self.combo_src.addItems(["auto", "en", "zh", "ja", "ko", "es", "fr", "de", "ru"])
        self.combo_src.setCurrentText(
            self.cm.get("translation").get("source_lang", "auto")
        )

        self.combo_target = QComboBox()
        self.combo_target.addItems(list(TARGET_LANGUAGE_CODES.keys()))
        self.combo_target.setCurrentText(
            self.cm.get("translation").get("target_lang", "zh-CN")
        )

        self.combo_src.currentTextChanged.connect(self.save_lang_settings)
        self.combo_target.currentTextChanged.connect(self.save_lang_settings)

        lang_layout.addWidget(QLabel("源语言:"))
        lang_layout.addWidget(self.combo_src)
        lang_layout.addWidget(QLabel("目标语言:"))
        lang_layout.addWidget(self.combo_target)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        # --- 外观设置 ---
        ui_group = QGroupBox("外观")
        ui_layout = QVBoxLayout()

        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(
            int(self.cm.get("display").get("opacity", 0.7) * 100)
        )
        self.slider_opacity.valueChanged.connect(self.update_appearance)

        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(12, 72)
        self.spin_font_size.setValue(
            self.cm.get("display").get("font_size", 24)
        )
        self.spin_font_size.valueChanged.connect(self.update_appearance)

        ui_layout.addWidget(QLabel("透明度:"))
        ui_layout.addWidget(self.slider_opacity)
        ui_layout.addWidget(QLabel("字体大小:"))
        ui_layout.addWidget(self.spin_font_size)
        ui_group.setLayout(ui_layout)
        layout.addWidget(ui_group)

        # --- 日志 ---
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(QLabel("日志:"))
        layout.addWidget(self.log_area)

    def load_model(self):
        """加载 SeamlessM4T 模型"""
        self.lbl_model_status.setText("正在加载模型...")
        self.progress_bar.show()
        self.btn_start.setEnabled(False)

        self.model_loader = ModelLoaderThread(self.translator)
        self.model_loader.progress.connect(self.on_model_progress)
        self.model_loader.finished.connect(self.on_model_loaded)
        self.model_loader.start()

    def on_model_progress(self, message: str):
        """模型加载进度回调"""
        self.lbl_model_status.setText(message)
        self.log(message)

    def on_model_loaded(self, success: bool, message: str):
        """模型加载完成回调"""
        self.progress_bar.hide()
        self.lbl_model_status.setText(message)
        self.log(message)

        if success:
            self.btn_start.setEnabled(True)
            self.lbl_model_status.setStyleSheet("color: green;")
        else:
            self.lbl_model_status.setStyleSheet("color: red;")

    def refresh_devices(self):
        """刷新音频设备列表"""
        self.combo_devices.clear()
        devices = self.audio_capture.list_devices()

        default_index = self.cm.get("audio").get("device_index")
        current_idx = 0

        for i, dev in enumerate(devices):
            name = f"[{dev['index']}] {dev['name']} ({dev['hostapi']})"
            self.combo_devices.addItem(name, dev['index'])
            if default_index is not None and dev['index'] == default_index:
                current_idx = i

        self.combo_devices.setCurrentIndex(current_idx)

    def save_audio_settings(self):
        """保存音频设置"""
        idx = self.combo_devices.currentData()
        if idx is not None:
            self.cm.set("audio", "device_index", idx)

    def save_lang_settings(self):
        """保存语言设置"""
        self.cm.set("translation", "source_lang", self.combo_src.currentText())
        self.cm.set("translation", "target_lang", self.combo_target.currentText())

    def update_appearance(self):
        """更新外观设置"""
        opacity = self.slider_opacity.value() / 100.0
        font_size = self.spin_font_size.value()

        self.cm.set("display", "opacity", opacity)
        self.cm.set("display", "font_size", font_size)

        # 更新字幕窗口
        display_config = self.cm.get("display")
        self.subtitle_window.config = display_config
        self.subtitle_window.update_styles()

    def toggle_translation(self):
        """切换翻译状态"""
        if self.worker and self.worker.isRunning():
            # 停止翻译
            self.worker.stop()
            self.worker.wait()
            self.btn_start.setText("开始翻译")
            self.btn_start.setStyleSheet(
                "background-color: #38A169; color: white; "
                "font-weight: bold; padding: 10px;"
            )
            self.log("翻译已停止")
        else:
            # 开始翻译
            self.worker = TranslationWorker(
                self.translator,
                self.audio_capture,
                self.cm
            )
            self.worker.text_updated.connect(self.subtitle_window.update_text)
            self.worker.status_updated.connect(self.log)
            self.worker.error_occurred.connect(lambda e: self.log(f"错误: {e}"))
            self.worker.start()

            self.btn_start.setText("停止翻译")
            self.btn_start.setStyleSheet(
                "background-color: #E53E3E; color: white; "
                "font-weight: bold; padding: 10px;"
            )
            self.log("翻译已开始")

    def log(self, message: str):
        """添加日志"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_area.append(f"[{timestamp}] {message}")
        # 自动滚动
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.subtitle_window.close()
        event.accept()
```

### 5. 入口文件 (main.py)

```python
"""
Seamless 实时翻译应用入口
"""
import sys
import os

def main():
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("警告: CUDA 不可用，将使用 CPU（性能会很慢）")
    except ImportError:
        print("错误: 未安装 PyTorch")
        sys.exit(1)

    from PyQt6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)

    # 设置字体
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

### 6. 配置管理器 (src/core/config_manager.py)

保留现有实现，但更新默认配置：

```python
"""
配置管理器
"""
import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "audio": {
        "device_index": None,
        "sample_rate": 16000
    },
    "translation": {
        "source_lang": "auto",
        "target_lang": "zh-CN"
    },
    "display": {
        "opacity": 0.7,
        "font_size": 24,
        "show_original": True
    },
    "model": {
        "name": "facebook/seamless-m4t-v2-large",
        "device": "cuda",
        "dtype": "float16"
    }
}

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> dict:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # 合并默认配置
                    return self._merge_config(DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"配置加载失败: {e}")
        return DEFAULT_CONFIG.copy()

    def _merge_config(self, default: dict, loaded: dict) -> dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"配置保存失败: {e}")

    def get(self, section: str, key: str = None, default=None):
        if key is None:
            return self.config.get(section, default if default is not None else {})
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
```

---

## 打包方案

### PyInstaller 打包脚本 (build.py)

```python
"""
PyInstaller 打包脚本
注意：SeamlessM4T 模型很大，不建议打包进 exe
建议：打包程序本身，模型在首次运行时自动下载到用户目录
"""
import PyInstaller.__main__
import os
import shutil

def build():
    # 清理
    for folder in ['dist', 'build']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    print("开始打包...")

    PyInstaller.__main__.run([
        'main.py',
        '--name=SeamlessTranslator',
        '--windowed',  # GUI 模式
        '--onedir',
        '--clean',

        # 收集依赖
        '--collect-all=transformers',
        '--collect-all=torch',
        '--collect-all=torchaudio',
        '--collect-all=sounddevice',
        '--collect-all=sentencepiece',

        # 隐藏导入
        '--hidden-import=PyQt6',
        '--hidden-import=numpy',

        # 路径
        '--paths=.',

        # 排除不需要的模块
        '--exclude-module=matplotlib',
        '--exclude-module=PIL',
        '--exclude-module=tkinter',
    ])

    print("打包完成！")
    print("注意：首次运行时会自动下载 SeamlessM4T 模型（约 10GB）")

if __name__ == "__main__":
    build()
```

### 打包注意事项

1. **不要打包模型**：SeamlessM4T v2 Large 模型约 10GB，不适合打包进 exe
2. **模型缓存位置**：HuggingFace 默认缓存到 `~/.cache/huggingface/`
3. **首次运行**：需要网络连接下载模型
4. **离线使用**：可以预先下载模型到指定目录，修改代码加载本地模型

---

## 部署检查清单

1. [ ] 安装 NVIDIA 驱动（最新版）
2. [ ] 确认 CUDA 版本要求：
   - **RTX 50 系列 (5070/5080/5090 Blackwell SM_120)**：需要 CUDA 12.8+
   - RTX 40/30 系列：CUDA 12.1+ 即可
3. [ ] 安装 Python 3.10+
4. [ ] 安装 PyTorch with CUDA：
   ```bash
   # RTX 50 系列 (Blackwell)
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

   # RTX 40/30 系列
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
5. [ ] 安装 bitsandbytes（Windows 需要特殊处理）：
   ```bash
   # 方法1: 使用 pip 直接安装（需要 bitsandbytes 0.43.0+ 支持 Windows）
   pip install bitsandbytes>=0.43.0

   # 方法2: 如果上面失败，使用预编译版本
   pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.43.0-py3-none-win_amd64.whl
   ```
6. [ ] 安装其他依赖：`pip install transformers accelerate sentencepiece sounddevice PyQt6`
7. [ ] 验证 CUDA 是否正常：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该输出 True
   print(torch.cuda.get_device_name(0))  # 应该显示 RTX 5070
   ```
8. [ ] 首次运行下载模型（约 10GB）
9. [ ] 测试音频采集功能
10. [ ] 测试翻译功能
11. [ ] 打包测试（可选）

---

## 已知限制

1. **显存需求**：
   - float16 模式：需要约 10GB 显存
   - **8-bit 量化模式：需要约 5-6GB 显存**（推荐 8GB 显存显卡使用）
   - 支持 RTX 5070 Laptop 8GB、RTX 4060 8GB 等显卡
2. **延迟**：端到端翻译比分离的 ASR+MT 方案延迟略高
3. **不支持真正的流式**：需要分块处理，无法做到同声传译级别的实时性
4. **模型下载**：首次使用需要下载约 10GB 模型

---

## 参考资源

- [SeamlessM4T v2 HuggingFace](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [Seamless Communication GitHub](https://github.com/facebookresearch/seamless_communication)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t_v2)

---

## 更新日志

- **v2.0** - 基于 SeamlessM4T v2 完全重构
- **v1.x** - 基于 faster_whisper + 翻译 API（已废弃）
