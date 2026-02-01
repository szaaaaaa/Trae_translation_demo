import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "audio": {
        "device_index": None,
        "sample_rate": 16000,
        "channels": 1
    },
    "recognition": {
        "model_size": "base",
        "device": "cpu",
        "compute_type": "int8"
    },
    "translation": {
        "source_lang": "auto",
        "target_lang": "zh-CN",
        "provider": "google"
    },
    "display": {
        "font_size": 24,
        "font_color": "#FFFFFF",
        "bg_color": "#000000",
        "opacity": 0.7,
        "position_x": 100,
        "position_y": 100,
        "width": 800,
        "height": 200
    }
}

class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Merge with default config to ensure all keys exist
                return self._merge_config(DEFAULT_CONFIG.copy(), loaded_config)
        except Exception as e:
            print(f"Error loading config: {e}")
            return DEFAULT_CONFIG.copy()

    def _merge_config(self, default, loaded):
        for key, value in default.items():
            if key in loaded:
                if isinstance(value, dict) and isinstance(loaded[key], dict):
                    self._merge_config(value, loaded[key])
                else:
                    default[key] = loaded[key]
        return default

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, section, key=None, default=None):
        if key is None:
            # Return entire section
            return self.config.get(section, default if default is not None else {})
        return self.config.get(section, {}).get(key, default)

    def set(self, section, key, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

if __name__ == "__main__":
    cm = ConfigManager()
    print("Config loaded:", cm.config)
    cm.save_config()
