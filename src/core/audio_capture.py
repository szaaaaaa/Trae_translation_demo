import sounddevice as sd
import numpy as np
import queue
import threading
import sys

class AudioCapture:
    def __init__(self, device_index=None, sample_rate=16000, channels=1):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None
        self._lock = threading.Lock()

    def list_devices(self):
        """List all available input devices, highlighting loopback ones."""
        devices = sd.query_devices()
        input_devices = []
        host_apis = sd.query_hostapis()
        
        for i, dev in enumerate(devices):
            # We are interested in input devices (max_input_channels > 0)
            # On Windows WASAPI, loopback devices appear as input devices.
            if dev['max_input_channels'] > 0:
                api_name = host_apis[dev['hostapi']]['name']
                is_loopback = False
                
                # Heuristic for Windows WASAPI loopback
                if 'WASAPI' in api_name:
                    # Often loopback devices don't have explicit "loopback" in name in some sd versions,
                    # but usually user can identify them by name (e.g. "Speakers (Realtek...)")
                    # However, purely "Speakers" is usually output. 
                    # If it shows as input, it might be a mic or loopback.
                    pass
                
                input_devices.append({
                    'index': i,
                    'name': dev['name'],
                    'hostapi': api_name,
                    'channels': dev['max_input_channels']
                })
        return input_devices

    def find_default_loopback_device(self):
        """Try to find a default loopback device on Windows WASAPI."""
        if sys.platform != 'win32':
            return None # Loopback is tricky on other platforms without specific setup
            
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        
        # 1. Look for WASAPI host API
        wasapi_index = -1
        for i, api in enumerate(host_apis):
            if 'WASAPI' in api['name']:
                wasapi_index = i
                break
        
        if wasapi_index == -1:
            return None
            
        # 2. Find the default output device for WASAPI
        default_output = host_apis[wasapi_index]['default_output_device']
        if default_output < 0:
            return None
            
        # 3. In sounddevice with WASAPI, the loopback device for a given output 
        # is often just the output device itself opened as an input stream?
        # Actually, sounddevice documentation says: 
        # "On Windows, 'WASAPI' devices support loopback mode if they are opened as input devices."
        # So we just need the index of the default OUTPUT device, but use it as INPUT.
        return default_output

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        if self.is_running:
            self.audio_queue.put(indata.copy())

    def start(self):
        if self.is_running:
            return

        with self._lock:
            try:
                if self.device_index is None:
                    # Try to auto-detect loopback on Windows
                    self.device_index = self.find_default_loopback_device()
                    if self.device_index is not None:
                         print(f"Auto-selected device index {self.device_index} for loopback.")
                
                # Check if device supports input
                if self.device_index is not None:
                     dev_info = sd.query_devices(self.device_index)
                     # Special case for WASAPI loopback: we open an OUTPUT device as INPUT.
                     # So we don't strictly check max_input_channels if it's WASAPI
                     pass

                self.is_running = True
                self.stream = sd.InputStream(
                    device=self.device_index,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=self._callback,
                    dtype=np.float32,
                    blocksize=int(self.sample_rate * 0.5) # 0.5s blocks
                )
                self.stream.start()
                print("Audio capture started.")
            except Exception as e:
                print(f"Failed to start audio capture: {e}")
                self.is_running = False
                raise e

    def stop(self):
        if not self.is_running:
            return
            
        with self._lock:
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            print("Audio capture stopped.")

    def get_data(self):
        """Generator that yields audio chunks."""
        while self.is_running:
            try:
                yield self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue

if __name__ == "__main__":
    # Test script
    capture = AudioCapture()
    print("Available Devices:")
    for dev in capture.list_devices():
        print(f"Index {dev['index']}: {dev['name']} ({dev['hostapi']})")
    
    # Attempt to start (might fail if no loopback found/selected)
    # capture.start()
    # import time
    # time.sleep(2)
    # capture.stop()
