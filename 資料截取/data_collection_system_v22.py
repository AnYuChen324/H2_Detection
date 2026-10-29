# -*- coding: utf-8 -*-
"""
數據收集系統 v17 - 真同步雙通道 transition 3ch 輸出版

新增功能：
1. ✅ 波形圖形化顯示（matplotlib）
2. ✅ 即時頻譜分析（FFT + STFT）
3. ✅ 收集進度條
4. ✅ 完善錯誤處理

@author: Paul Wu
@date: 2025-02-05
@version: 17
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
import queue
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import json
import shutil
import traceback
import csv
import re
import sys

PROGRAM_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(PROGRAM_DIR) not in sys.path:
    sys.path.insert(0, str(PROGRAM_DIR))

# Matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "MingLiU",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
plt.ioff()  # 關閉交互模式，防止彈出視窗
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 科學計算
from scipy import ndimage, signal
from scipy.fft import fft, fftfreq

# 小波去噪（選配；未安裝時自動停用該選項）
try:
    import pywt
    PYWT_AVAILABLE = True
except Exception:
    pywt = None
    PYWT_AVAILABLE = False

# TDMS
try:
    from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
    TDMS_AVAILABLE = True
except ImportError:
    TdmsFile = None
    TDMS_AVAILABLE = False
    print("[WARN] nptdms未安裝")

# NI-DAQ
NIDAQ_IMPORT_ERROR = ""

NIDAQ_CAPTURE_PATH = str(PROGRAM_DIR / "nidaq_capture_v2.py")
try:
    from nidaq_capture_v2 import (
        AEsignal_Capture,
        AEsignal_MultiChannelCapture,
        AEsignal_DualChannelContinuousReader,
        ContinuousChunkRecorder,
        NIDAQMX_IMPORT_ERROR,
    )
    NIDAQ_AVAILABLE = not bool(NIDAQMX_IMPORT_ERROR)
    NIDAQ_IMPORT_ERROR = NIDAQMX_IMPORT_ERROR
except Exception as exc:
    NIDAQ_AVAILABLE = False
    AEsignal_Capture = None
    AEsignal_MultiChannelCapture = None
    AEsignal_DualChannelContinuousReader = None
    ContinuousChunkRecorder = None
    NIDAQMX_IMPORT_ERROR = ""
    NIDAQ_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[WARN] NI-DAQ unavailable ({NIDAQ_IMPORT_ERROR}); using simulation mode")

try:
    from leak_detectors import (AdaptiveThresholdDetector, CUSUMDetector,
                                  BurstRateDetector, ReferenceSubtractor, median_mad)
    LEAK_DETECTORS_AVAILABLE = True
except Exception as exc:
    LEAK_DETECTORS_AVAILABLE = False
    print(f"[WARN] leak_detectors unavailable ({type(exc).__name__}: {exc})")


# ------------------------------------------------------------
# 擷取量程 / 耦合 UI 選項與解析
# 背景：AE 訊號 ~±30mV 卻用 ±10V 量程收 → 微洩漏低於量化階(LSB≈0.32mV)看不到。
# 窄量程提升解析度；但需先以 AC 耦合移除 ~-0.6V DC，否則窄量程會削波。
# ------------------------------------------------------------
ACQ_RANGE_OPTIONS = [
    ("±10 V (預設)", 10.0),
    ("±5 V", 5.0),
    ("±2 V", 2.0),
    ("±1 V", 1.0),
    ("±0.5 V", 0.5),
    ("±0.2 V", 0.2),
    ("±0.1 V", 0.1),
]
ACQ_COUPLING_OPTIONS = ["DC (預設)", "AC"]
ACQ_TERMINAL_OPTIONS = ["預設", "DIFF", "RSE", "NRSE", "PSEUDO_DIFF"]


def parse_acq_controls(range_label, coupling_label, terminal_label):
    """把 UI 選項字串轉成 reader 的 kwargs：min_val/max_val/coupling/terminal_config。"""
    rng = next((v for lbl, v in ACQ_RANGE_OPTIONS if lbl == range_label), 10.0)
    coupling = "AC" if str(coupling_label).strip().upper().startswith("AC") else None
    terminal = None if (not terminal_label or terminal_label == "預設") else terminal_label
    return {"min_val": -rng, "max_val": rng, "coupling": coupling, "terminal_config": terminal}


# ============================================================
# 錯誤處理模組
# ============================================================

class ErrorHandler:
    """錯誤處理器"""
    
    @staticmethod
    def validate_params(sample_rate, sample_num, duration):
        """驗證採樣參數"""
        errors = []
        warnings = []
        
        # 採樣率
        if sample_rate < 1000:
            errors.append("採樣率太低（建議 ≥ 10,000 Hz）")
        elif sample_rate > 10000000:
            errors.append("採樣率太高（建議 ≤ 1,000,000 Hz）")
        elif sample_rate > 1000000:
            warnings.append("採樣率較高，注意記憶體使用")
        
        # 採樣點數
        if sample_num < 100:
            errors.append("採樣點數太少（建議 ≥ 1,000）")
        elif sample_num > 100000000:
            errors.append("採樣點數太多（可能記憶體不足）")
        
        # 時間
        if duration < 0.001:
            errors.append("擷取時間太短（建議 ≥ 0.01 秒）")
        elif duration > 3600:
            warnings.append("擷取時間很長（≥ 1 小時）")
        
        # 記憶體估算
        memory_mb = sample_num * 4 * 8 / 1024 / 1024
        if memory_mb > 1000:
            warnings.append(f"預估記憶體使用 {memory_mb:.0f} MB")
        
        if errors:
            return False, "❌ 參數錯誤:\n• " + "\n• ".join(errors)
        elif warnings:
            return True, "⚠ 警告:\n• " + "\n• ".join(warnings)
        
        return True, "✓ 參數正常"
    
    @staticmethod
    def check_disk_space(folder, required_mb):
        """檢查磁碟空間"""
        try:
            folder_path = Path(folder)
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
            
            stat = shutil.disk_usage(folder)
            available_mb = stat.free / 1024 / 1024
            
            if available_mb < required_mb:
                return False, f"❌ 磁碟空間不足\n需要: {required_mb:.0f} MB\n剩餘: {available_mb:.0f} MB"
            
            if available_mb < required_mb * 2:
                return True, f"⚠ 磁碟空間偏少\n剩餘: {available_mb:.0f} MB"
            
            return True, f"✓ 磁碟空間充足\n剩餘: {available_mb:.0f} MB"
        
        except Exception as e:
            return False, f"❌ 無法檢查磁碟空間: {e}"
    
    @staticmethod
    def log_error(error, context=""):
        """記錄錯誤到日誌"""
        try:
            log_file = Path("error.log")
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"時間: {datetime.now()}\n")
                f.write(f"情境: {context}\n")
                f.write(f"錯誤: {str(error)}\n")
                f.write(f"{'-'*70}\n")
                f.write(traceback.format_exc())
                f.write(f"{'='*70}\n")
        except:
            pass


class SmartRetry:
    """智能重試機制"""
    
    @staticmethod
    def retry_with_backoff(func, max_retries=3, backoff_factor=1.5, 
                          on_retry=None):
        """指數退避重試"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = backoff_factor ** attempt
                
                if on_retry:
                    on_retry(attempt + 1, max_retries, wait_time, e)
                
                time.sleep(wait_time)


# ============================================================
# 單通道 near/far session 管理
# ============================================================

class ConditionFolderHelper:
    """解析並準備既有條件母資料夾。"""

    CONDITION_RE = re.compile(
        r"^(?:(?P<volume>\d+)_(?P<leak_kind>leak)-near|(?P<normal_kind>normal)-near)\s+(?P<distance>\d+)\s*cm$",
        re.IGNORECASE,
    )

    @staticmethod
    def parse(folder):
        path = Path(folder)
        match = ConditionFolderHelper.CONDITION_RE.match(path.name.strip())
        if not match:
            return None

        kind = "leak" if match.group("leak_kind") else "normal"
        volume = int(match.group("volume")) if kind == "leak" else 0
        return {
            "matched_volume": volume,
            "leak_status": "leak1" if kind == "leak" else "leak0",
            "volume_ml": volume,
            "position_type": "straight",
            "distance_cm": int(match.group("distance")),
            "condition_folder": path.name,
        }

    @staticmethod
    def ensure_session_dirs(folder):
        base = Path(folder)
        dirs = {
            "near": base / "near",
            "far": base / "far",
            "merged_3ch": base / "merged_3ch",
            "ai0": base / "ai0",
            "ai1": base / "ai1",
            "dual_2ch": base / "dual_2ch",
            "transition_dual": base / "transition_dual",
            "transition_dual_ai0": base / "transition_dual" / "ai0",
            "transition_dual_ai1": base / "transition_dual" / "ai1",
            "transition_dual_merged": base / "transition_dual" / "merged_3ch",
            "staging": base / "_session_staging",
        }
        for subdir in dirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        return dirs


class StageIO:
    """輕量暫存格式，避免長時間採集全部壓在記憶體。"""

    @staticmethod
    def save(path, signal_data, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, signal=np.asarray(signal_data, dtype=np.float32),
                 metadata=json.dumps(metadata, ensure_ascii=False))
        return path

    @staticmethod
    def load(path):
        with np.load(str(path), allow_pickle=False) as payload:
            signal_data = payload["signal"].astype(np.float32)
            metadata_raw = payload["metadata"]
            if hasattr(metadata_raw, "item"):
                metadata_raw = metadata_raw.item()
            metadata = json.loads(str(metadata_raw))
        return signal_data, metadata

    @staticmethod
    def save_dual(path, data_dict, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        channels = list(data_dict.keys())
        payload = {
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "channels": json.dumps(channels, ensure_ascii=False),
        }
        for channel in channels:
            payload[channel] = np.asarray(data_dict[channel], dtype=np.float32)
        np.savez(path, **payload)
        return path

    @staticmethod
    def load_dual(path):
        with np.load(str(path), allow_pickle=False) as payload:
            metadata_raw = payload["metadata"]
            if hasattr(metadata_raw, "item"):
                metadata_raw = metadata_raw.item()
            metadata = json.loads(str(metadata_raw))

            channels_raw = payload["channels"] if "channels" in payload.files else "[]"
            if hasattr(channels_raw, "item"):
                channels_raw = channels_raw.item()
            channels = json.loads(str(channels_raw))
            if not channels:
                channels = [name for name in payload.files if name not in ("metadata", "channels")]

            data_dict = {
                channel: payload[channel].astype(np.float32)
                for channel in channels
                if channel in payload.files
            }
        return data_dict, metadata

    @staticmethod
    def load_metadata(path):
        with np.load(str(path), allow_pickle=False) as payload:
            metadata_raw = payload["metadata"]
            if hasattr(metadata_raw, "item"):
                metadata_raw = metadata_raw.item()
            return json.loads(str(metadata_raw))


# ============================================================
# 多通道擷取管理器
# ============================================================

class MultiChannelCapture:
    """多通道同步擷取"""
    
    def __init__(self):
        self.channels = []
        self.device_name = "cDAQ1Mod1"
        self.simulation_mode = False
    
    def configure_device(self, device_name, simulation_mode=False):
        """配置設備名稱和模式"""
        self.device_name = device_name
        self.simulation_mode = simulation_mode
    
    def configure_channels(self, channel_configs):
        """配置通道"""
        self.channels = channel_configs
    
    def capture_sync(self, sample_rate, sample_num):
        """同步擷取所有通道"""
        captured_data = {}
        enabled_channels = [ch for ch in self.channels if ch.get('enabled')]
        
        # 強制模擬模式
        if self.simulation_mode:
            print("使用模擬模式")
            for ch_config in enabled_channels:
                ch_name = ch_config['id']
                captured_data[ch_name] = self._simulate_signal(sample_num)
            return captured_data
        
        # 實際擷取模式
        if not NIDAQ_AVAILABLE:
            detail = NIDAQ_IMPORT_ERROR or "unknown import error"
            raise Exception(
                "NI-DAQ 模組未載入，請確認 Spyder 使用的 Python 環境有安裝 nidaqmx，"
                f"且可讀取 {NIDAQ_CAPTURE_PATH}\n原因: {detail}"
            )
        
        if not enabled_channels:
            raise Exception("No enabled channels")

        if len(enabled_channels) > 1:
            channel_pairs = []
            for ch_config in enabled_channels:
                ch_name = ch_config['id']
                full_channel_name = ch_name if "/" in ch_name else f"{self.device_name}/{ch_name}"
                channel_pairs.append((ch_name, full_channel_name))

            try:
                full_channel_names = [full for _, full in channel_pairs]
                print(f"Synchronously capturing channels: {', '.join(full_channel_names)}")
                capture = AEsignal_MultiChannelCapture(
                    samps=sample_num,
                    samplerate=sample_rate,
                    channelnames=full_channel_names,
                )
                data_by_full_channel, _ = capture.capture()
                for ch_name, full_channel_name in channel_pairs:
                    data = data_by_full_channel[full_channel_name]
                    captured_data[ch_name] = np.array(data)
                    print(f"{ch_name} synchronized capture complete ({len(data)} samples)")
            except Exception as e:
                channel_text = ", ".join(full for _, full in channel_pairs)
                error_msg = f"Channel capture failed for {channel_text}:\n{str(e)}"
                print(error_msg)
                raise Exception(error_msg)

            return captured_data

        for ch_config in enabled_channels:
            if ch_config['enabled']:
                ch_name = ch_config['id']
                full_channel_name = ch_name if "/" in ch_name else f"{self.device_name}/{ch_name}"
                
                try:
                    print(f"擷取通道: {full_channel_name}")
                    capture = AEsignal_Capture(
                        samps=sample_num,
                        samplerate=sample_rate,
                        channelname=full_channel_name,
                        savecheck="0",
                        savepath="./"
                    )
                    data, _ = capture.capture()
                    captured_data[ch_name] = np.array(data)
                    print(f"✓ {ch_name} 擷取成功 ({len(data)} 點)")
                    
                except Exception as e:
                    error_msg = f"❌ 通道 {full_channel_name} 擷取失敗:\n{str(e)}"
                    print(error_msg)
                    raise Exception(error_msg)
        
        if not captured_data:
            raise Exception("❌ 沒有成功擷取任何通道數據")
        
        return captured_data
    
    def _simulate_signal(self, sample_num):
        """生成模擬訊號"""
        signal = np.random.randn(sample_num) * 0.001
        
        event_start = np.random.randint(1000, max(1001, sample_num - 10000))
        event_length = min(8000, sample_num - event_start - 100)
        
        if event_length > 100:
            t = np.linspace(0, 0.02, event_length)
            freq = 50000 + np.random.randn() * 5000
            envelope = np.exp(-t * 500)
            ae_event = envelope * np.sin(2 * np.pi * freq * t) * 0.15
            signal[event_start:event_start+event_length] += ae_event
        
        return signal


class TDMSSaver:
    """TDMS儲存管理"""

    @staticmethod
    def _sanitize_metadata(metadata):
        """TDMS properties only support simple scalar values."""
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                clean[key] = value.item()
            else:
                clean[key] = str(value)
        return clean
    
    @staticmethod
    def save(filepath, captured_data, metadata):
        """儲存TDMS檔案"""
        if not TDMS_AVAILABLE:
            return TDMSSaver._save_numpy(filepath, captured_data, metadata)
        
        try:
            metadata = TDMSSaver._sanitize_metadata(metadata)
            root_object = RootObject(properties=metadata)
            group_object = GroupObject("AE_Signals")
            
            channels = []
            for ch_name, data in captured_data.items():
                channel = ChannelObject(
                    "AE_Signals",
                    ch_name,
                    data,
                    properties={
                        "unit": "V",
                        "sample_rate": metadata.get('sample_rate', 0)
                    }
                )
                channels.append(channel)
            
            with TdmsWriter(str(filepath)) as writer:
                writer.write_segment([root_object, group_object] + channels)
            
            return True
            
        except Exception as e:
            print(f"TDMS儲存失敗: {e}")
            return TDMSSaver._save_numpy(filepath, captured_data, metadata)
    
    @staticmethod
    def _save_numpy(filepath, captured_data, metadata):
        """備用：NumPy格式"""
        try:
            filepath_npz = str(filepath).replace('.tdms', '.npz')
            save_dict = {
                'metadata': json.dumps(metadata),
                **captured_data
            }
            np.savez(filepath_npz, **save_dict)
            return True
        except Exception as e:
            print(f"NumPy儲存失敗: {e}")
            return False


# ============================================================
# 波形顯示器（matplotlib）
# ============================================================

class WaveformViewer:
    """輕量頻寬波形顯示器。"""

    MAX_PLOT_POINTS = 6000

    def __init__(self, parent_frame):
        control_frame = ttk.Frame(parent_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=3)

        ttk.Label(control_frame, text="頻寬").pack(side=tk.LEFT, padx=(2, 3))
        self.low_hz_var = tk.StringVar(value="5000")
        self.high_hz_var = tk.StringVar(value="170000")
        self.low_hz_entry = ttk.Entry(control_frame, textvariable=self.low_hz_var, width=8)
        self.low_hz_entry.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="到").pack(side=tk.LEFT, padx=3)
        self.high_hz_entry = ttk.Entry(control_frame, textvariable=self.high_hz_var, width=8)
        self.high_hz_entry.pack(side=tk.LEFT)
        ttk.Label(control_frame, text="Hz").pack(side=tk.LEFT, padx=3)
        ttk.Button(control_frame, text="更新", command=self.plot_normal).pack(side=tk.LEFT, padx=6)
        ttk.Button(control_frame, text="疊加", command=self.plot_overlay).pack(side=tk.LEFT, padx=3)
        ttk.Button(control_frame, text="聚焦事件", command=self.zoom_to_event).pack(side=tk.LEFT, padx=3)

        self.fig = Figure(figsize=(10, 6), dpi=90)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax_raw = self.fig.add_subplot(3, 1, 1)
        self.ax_band = self.fig.add_subplot(3, 1, 2)
        self.ax_stats = self.fig.add_subplot(3, 1, 3)
        self.axes = [self.ax_raw, self.ax_band, self.ax_stats]

        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(parent_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        self.current_data = None
        self.current_sample_rate = None
        self.display_mode = 'normal'
        self.channel_cache = {}

    def _band_limits(self, sample_rate):
        nyquist = sample_rate / 2.0
        try:
            low_hz = float(self.low_hz_var.get())
            high_hz = float(self.high_hz_var.get())
        except Exception:
            low_hz, high_hz = 5000.0, 170000.0
        low_hz = max(0.0, low_hz)
        high_hz = min(max(high_hz, low_hz + 1.0), nyquist * 0.98)
        return low_hz, high_hz

    def _downsample(self, data, sample_rate):
        data = np.asarray(data)
        if data.size <= self.MAX_PLOT_POINTS:
            index = np.arange(data.size)
        else:
            step = int(np.ceil(data.size / self.MAX_PLOT_POINTS))
            index = np.arange(0, data.size, step)
        return index / sample_rate * 1000.0, data[index]

    def _bandpass(self, raw, sample_rate, low_hz, high_hz):
        raw = np.asarray(raw, dtype=np.float64)
        dc_offset = float(np.mean(raw)) if raw.size else 0.0
        ac = raw - dc_offset
        if raw.size < 16:
            return dc_offset, ac.astype(np.float32)
        try:
            if low_hz <= 0:
                sos = signal.butter(4, high_hz, btype='lowpass', fs=sample_rate, output='sos')
            else:
                sos = signal.butter(4, [low_hz, high_hz], btype='bandpass', fs=sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, ac)
        except Exception:
            filtered = ac
        return dc_offset, filtered.astype(np.float32)

    def _features(self, band_data, sample_rate, low_hz, high_hz):
        band_data = np.asarray(band_data, dtype=np.float64)
        if band_data.size == 0:
            return {
                "rms": 0.0,
                "peak": 0.0,
                "main_freq": 0.0,
                "centroid": 0.0,
                "bands": [],
            }

        rms = float(np.sqrt(np.mean(band_data ** 2)))
        peak = float(np.max(np.abs(band_data)))

        nperseg = min(32768, max(256, band_data.size))
        freqs, psd = signal.welch(band_data, fs=sample_rate, nperseg=nperseg, noverlap=0)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        f = freqs[mask]
        p = psd[mask]
        total = float(np.sum(p))
        if total > 0 and f.size:
            main_freq = float(f[int(np.argmax(p))])
            centroid = float(np.sum(f * p) / total)
        else:
            main_freq = 0.0
            centroid = 0.0

        edges = [
            (low_hz, min(20000.0, high_hz), "5-20k"),
            (max(20000.0, low_hz), min(50000.0, high_hz), "20-50k"),
            (max(50000.0, low_hz), min(100000.0, high_hz), "50-100k"),
            (max(100000.0, low_hz), high_hz, "100-170k"),
        ]
        bands = []
        for lo, hi, label in edges:
            if hi <= lo or total <= 0:
                ratio = 0.0
            else:
                ratio = float(np.sum(psd[(freqs >= lo) & (freqs < hi)]) / total)
            bands.append((label, ratio))

        return {
            "rms": rms,
            "peak": peak,
            "main_freq": main_freq,
            "centroid": centroid,
            "bands": bands,
        }

    def _channel_order(self, data_dict):
        preferred = [ch for ch in ("ai0", "ai1", "ai2", "ai3") if ch in data_dict]
        others = [ch for ch in data_dict.keys() if ch not in preferred]
        return preferred + others

    def _prepare_channels(self, data_dict, sample_rate):
        low_hz, high_hz = self._band_limits(sample_rate)
        prepared = {}
        for ch_name in self._channel_order(data_dict)[:4]:
            raw = np.asarray(data_dict[ch_name], dtype=np.float32)
            dc_offset, band = self._bandpass(raw, sample_rate, low_hz, high_hz)
            prepared[ch_name] = {
                "raw": raw,
                "dc": dc_offset,
                "band": band,
                "features": self._features(band, sample_rate, low_hz, high_hz),
            }
        self.channel_cache = prepared
        return low_hz, high_hz, prepared

    def plot_waveforms(self, data_dict, sample_rate):
        """顯示 raw 與扣 DC + bandpass 後的雙通道波形。"""
        self.current_data = data_dict
        self.current_sample_rate = sample_rate
        self.display_mode = 'normal'

        low_hz, high_hz, prepared = self._prepare_channels(data_dict, sample_rate)
        self.fig.clear()
        self.ax_raw = self.fig.add_subplot(3, 1, 1)
        self.ax_band = self.fig.add_subplot(3, 1, 2)
        self.ax_stats = self.fig.add_subplot(3, 1, 3)
        self.axes = [self.ax_raw, self.ax_band, self.ax_stats]
        colors = {
            "ai0": "#1f77b4",
            "ai1": "#d62728",
            "ai2": "#2ca02c",
            "ai3": "#9467bd",
        }

        for ax in self.axes:
            ax.clear()
            ax.set_facecolor('#ffffff')

        for ch_name, info in prepared.items():
            color = colors.get(ch_name, "#444444")
            t_raw, raw_plot = self._downsample(info["raw"], sample_rate)
            t_band, band_plot = self._downsample(info["band"], sample_rate)
            self.ax_raw.plot(t_raw, raw_plot, color=color, linewidth=0.7, alpha=0.75,
                             label=f"{ch_name} raw (DC {info['dc']:.4f} V)")
            self.ax_band.plot(t_band, band_plot * 1000.0, color=color, linewidth=0.7, alpha=0.85,
                              label=f"{ch_name} band")

        self.ax_raw.set_title("Raw voltage")
        self.ax_raw.set_ylabel("V")
        self.ax_raw.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        self.ax_raw.legend(loc='upper right', fontsize=8)

        self.ax_band.set_title(f"DC removed + bandpass {low_hz/1000:.1f}-{high_hz/1000:.1f} kHz")
        self.ax_band.set_ylabel("mV")
        self.ax_band.set_xlabel("時間 (ms)")
        self.ax_band.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        self.ax_band.legend(loc='upper right', fontsize=8)

        self.ax_stats.axis('off')
        stats_lines = []
        for ch_name, info in prepared.items():
            feat = info["features"]
            band_text = " | ".join(f"{label}:{ratio*100:.1f}%" for label, ratio in feat["bands"])
            stats_lines.append(
                f"{ch_name}  DC={info['dc']:.5f} V  "
                f"RMS={feat['rms']*1000:.3f} mV  "
                f"Peak={feat['peak']*1000:.3f} mV  "
                f"Main={feat['main_freq']/1000:.1f} kHz  "
                f"Centroid={feat['centroid']/1000:.1f} kHz\n"
                f"      Energy  {band_text}"
            )
        self.ax_stats.text(
            0.01, 0.96, "\n".join(stats_lines) if stats_lines else "No data",
            transform=self.ax_stats.transAxes,
            va='top',
            fontsize=9,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#cccccc'),
        )

        self.fig.tight_layout(pad=1.3)
        self.canvas.draw_idle()

    def plot_normal(self):
        if self.current_data is not None:
            self.plot_waveforms(self.current_data, self.current_sample_rate)

    def plot_overlay(self):
        if not self.current_data:
            return

        self.display_mode = 'overlay'
        low_hz, high_hz, prepared = self._prepare_channels(self.current_data, self.current_sample_rate)
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        self.axes = [ax]
        ax.set_facecolor('#ffffff')
        colors = {
            "ai0": "#1f77b4",
            "ai1": "#d62728",
            "ai2": "#2ca02c",
            "ai3": "#9467bd",
        }
        for ch_name, info in prepared.items():
            t_ms, band_plot = self._downsample(info["band"], self.current_sample_rate)
            ax.plot(t_ms, band_plot * 1000.0, color=colors.get(ch_name, "#444444"),
                    linewidth=0.8, alpha=0.8, label=ch_name)
        ax.set_title(f"Bandpass overlay {low_hz/1000:.1f}-{high_hz/1000:.1f} kHz")
        ax.set_xlabel("時間 (ms)")
        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        self.fig.tight_layout(pad=1.3)
        self.canvas.draw_idle()

    def zoom_to_event(self):
        if not self.channel_cache or self.current_sample_rate is None:
            return

        best = None
        for info in self.channel_cache.values():
            band = np.asarray(info.get("band", []))
            if band.size == 0:
                continue
            peak_idx = int(np.argmax(np.abs(band)))
            peak_value = float(abs(band[peak_idx]))
            if best is None or peak_value > best[0]:
                best = (peak_value, peak_idx)

        if best is None:
            return

        _, peak_idx = best
        window = int(max(1000, self.current_sample_rate * 0.025))
        start_ms = max(0, peak_idx - window) / self.current_sample_rate * 1000.0
        end_ms = (peak_idx + window) / self.current_sample_rate * 1000.0
        for ax in self.axes:
            if ax.has_data():
                ax.set_xlim(start_ms, end_ms)
        self.canvas.draw_idle()


# ============================================================
# 頻譜分析器
# ============================================================

class SpectrumAnalyzer:
    """頻譜分析器"""
    
    def __init__(self, parent_frame):
        # 創建Figure
        self.fig = Figure(figsize=(10, 8), dpi=90)
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # 創建3個子圖
        self.ax_fft = self.fig.add_subplot(3, 1, 1)
        self.ax_stft = self.fig.add_subplot(3, 1, 2)
        self.ax_stats = self.fig.add_subplot(3, 1, 3)
        
        self.fig.tight_layout(pad=2.0)
        
        # 嵌入tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 工具列
        toolbar_frame = ttk.Frame(parent_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def analyze(self, data_dict, sample_rate):
        """頻譜分析"""
        # 清空
        self.ax_fft.clear()
        self.ax_stft.clear()
        self.ax_stats.clear()
        
        # 選擇能量最大的通道
        max_energy = 0
        selected_ch = None
        
        for ch_name, data in data_dict.items():
            energy = np.sum(data**2)
            if energy > max_energy:
                max_energy = energy
                selected_ch = ch_name
        
        if not selected_ch:
            return
        
        data = data_dict[selected_ch]
        
        # === 1. FFT功率譜 ===
        n = len(data)
        fft_vals = fft(data)
        fft_freq = fftfreq(n, 1/sample_rate)
        
        # 只取正頻率
        positive_idx = fft_freq > 0
        freq = fft_freq[positive_idx]
        power = np.abs(fft_vals[positive_idx])**2
        power_db = 10 * np.log10(power + 1e-12)
        
        self.ax_fft.plot(freq/1000, power_db, 
                        color='#1f77b4', linewidth=0.8)
        self.ax_fft.set_xlabel('頻率 (kHz)', fontsize=10)
        self.ax_fft.set_ylabel('功率 (dB)', fontsize=10)
        self.ax_fft.set_title(f'功率頻譜 - {selected_ch}', 
                             fontsize=11, fontweight='bold')
        self.ax_fft.grid(True, alpha=0.3, linestyle='--')
        self.ax_fft.set_xlim(0, min(200, sample_rate/2000))
        
        # 找峰值
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(power, height=np.max(power)*0.1, 
                                      distance=100)
        
        if len(peaks) > 0:
            top_3_indices = np.argsort(power[peaks])[-3:][::-1]
            top_peaks = peaks[top_3_indices]
            
            for peak_idx in top_peaks:
                peak_freq = freq[peak_idx]
                peak_power_db = power_db[peak_idx]
                self.ax_fft.axvline(peak_freq/1000, color='red', 
                                   linestyle='--', alpha=0.5)
                self.ax_fft.plot(peak_freq/1000, peak_power_db, 
                                'ro', markersize=6)
                self.ax_fft.text(peak_freq/1000, peak_power_db + 5, 
                                f'{peak_freq/1000:.1f}kHz',
                                fontsize=8, color='red')
        
        # === 2. 時頻圖 (STFT) ===
        nperseg = min(2048, n//10)
        f, t, Zxx = signal.stft(data, sample_rate, nperseg=nperseg)
        
        im = self.ax_stft.pcolormesh(t*1000, f/1000, np.abs(Zxx), 
                                     shading='gouraud', cmap='viridis')
        self.ax_stft.set_xlabel('時間 (ms)', fontsize=10)
        self.ax_stft.set_ylabel('頻率 (kHz)', fontsize=10)
        self.ax_stft.set_title('時頻圖 (STFT)', fontsize=11, fontweight='bold')
        self.ax_stft.set_ylim(0, min(200, sample_rate/2000))
        
        # 添加colorbar
        try:
            self.fig.colorbar(im, ax=self.ax_stft, label='振幅')
        except:
            pass
        
        # === 3. 統計資訊 ===
        self.ax_stats.axis('off')
        
        # 計算特徵
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))
        crest_factor = peak / rms if rms > 0 else 0
        
        # 主頻率
        main_freq_idx = np.argmax(power) if len(power) > 0 else 0
        main_freq = freq[main_freq_idx] if len(freq) > 0 else 0
        
        # 頻帶能量
        low_band = np.sum(power[freq < 50000])
        mid_band = np.sum(power[(freq >= 50000) & (freq < 100000)])
        high_band = np.sum(power[freq >= 100000])
        total_power = low_band + mid_band + high_band
        
        # 訊號品質評估
        quality = "優秀" if crest_factor > 3 and rms > 0.01 else \
                  "良好" if crest_factor > 2 and rms > 0.005 else "可接受"
        
        is_typical = "是" if 40000 < main_freq < 60000 else "否"
        
        stats_text = f"""
╔══════════════════════════════════════════════════╗
║              頻譜分析報告 - {selected_ch}                ║
╚══════════════════════════════════════════════════╝

【訊號統計】
  • RMS 值:          {rms*1e3:.3f} mV
  • 峰值:            {peak*1e3:.3f} mV
  • 峰值因子:        {crest_factor:.2f}
  • 主頻率:          {main_freq/1000:.1f} kHz

【頻帶能量分佈】
  • 低頻 (<50kHz):   {low_band/total_power*100:.1f}%  {'▓'*int(low_band/total_power*20)}
  • 中頻 (50-100):   {mid_band/total_power*100:.1f}%  {'▓'*int(mid_band/total_power*20)}
  • 高頻 (>100kHz):  {high_band/total_power*100:.1f}%  {'▓'*int(high_band/total_power*20)}

【特徵評估】
  • 訊號品質:        {quality}
  • 典型AE頻率:      {is_typical}
  • TDOA適用性:      {'✓ 適合' if rms > 0.005 else '⚠ 訊號偏弱'}

【建議】
  {'✓ 訊號品質良好，可進行TDOA定位' if quality == '優秀' else '• 建議檢查sensor連接或增加增益'}
        """
        
        self.ax_stats.text(0.05, 0.95, stats_text, 
                          transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round,pad=0.8', 
                                  facecolor='#ffffcc', alpha=0.8,
                                  edgecolor='#cccccc'))
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()


# ============================================================
# 進度追蹤器
# ============================================================

class ProgressTracker:
    """進度追蹤器"""
    
    def __init__(self, parent_frame):
        self.frame = parent_frame
        
        # 總進度
        progress_frame = ttk.LabelFrame(self.frame, text="總體進度", padding=10)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.total_progress = ttk.Progressbar(progress_frame, 
                                              mode='determinate',
                                              length=500)
        self.total_progress.pack(fill='x', pady=5)
        
        self.total_label = ttk.Label(progress_frame, text="0/0 (0.0%)",
                                     font=('Arial', 10))
        self.total_label.pack()
        
        # 當前進度
        current_frame = ttk.LabelFrame(self.frame, text="當前擷取", padding=10)
        current_frame.pack(fill='x', padx=10, pady=5)
        
        self.current_progress = ttk.Progressbar(current_frame, 
                                               mode='indeterminate',
                                               length=500)
        self.current_progress.pack(fill='x', pady=5)
        
        self.current_label = ttk.Label(current_frame, text="就緒",
                                       font=('Arial', 9))
        self.current_label.pack()
        
        # 統計資訊
        stats_frame = ttk.LabelFrame(self.frame, text="統計資訊", padding=10)
        stats_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=50,
                                 font=('Courier', 9), bg='#f9f9f9')
        self.stats_text.pack(fill='both', expand=True)
        
        # 最近記錄
        recent_frame = ttk.LabelFrame(self.frame, text="最近記錄", padding=5)
        recent_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.recent_text = scrolledtext.ScrolledText(recent_frame, 
                                                     height=8, width=50,
                                                     font=('Courier', 8))
        self.recent_text.pack(fill='both', expand=True)
        
        # 初始化
        self.reset()
    
    def reset(self):
        """重置"""
        self.start_time = None
        self.total_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.total_progress['value'] = 0
        self.current_progress.stop()
        self.stats_text.delete(1.0, tk.END)
        self.recent_text.delete(1.0, tk.END)
    
    def start(self, total_count):
        """開始追蹤"""
        self.start_time = time.time()
        self.total_count = total_count
        self.success_count = 0
        self.fail_count = 0
        self.total_progress['maximum'] = 100
        self.total_progress['value'] = 0
        self.current_progress.start(10)
    
    def update(self, current_count, filename, success, file_size_mb=0):
        """更新進度"""
        # 更新計數
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1
        
        # 更新總進度
        progress = (current_count / self.total_count) * 100
        self.total_progress['value'] = progress
        self.total_label.config(
            text=f"{current_count}/{self.total_count} ({progress:.1f}%)"
        )
        
        # 計算統計
        elapsed = time.time() - self.start_time
        rate = current_count / elapsed if elapsed > 0 else 0
        remaining = (self.total_count - current_count) / rate if rate > 0 else 0
        
        success_rate = (self.success_count / current_count * 100) if current_count > 0 else 0
        
        # 更新統計資訊
        stats = f"""
╔════════════════════════════════════════╗
║           收集統計                      ║
╚════════════════════════════════════════╝

已收集:     {current_count} / {self.total_count} 筆
成功:       {self.success_count} 筆 ({success_rate:.1f}%)
失敗:       {self.fail_count} 筆

已用時間:   {self._format_time(elapsed)}
預計剩餘:   {self._format_time(remaining)}
平均速度:   {rate*60:.1f} 筆/分

當前狀態:   {'✓ 成功' if success else '✗ 失敗'}
        """
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
        
        # 更新最近記錄
        status = "✓" if success else "✗"
        size_str = f"({file_size_mb:.1f}MB)" if file_size_mb > 0 else ""
        record = f"{status} {current_count:3d}. {filename} {size_str}\n"
        
        self.recent_text.insert(1.0, record)
        
        # 只保留最近15筆
        lines = self.recent_text.get(1.0, tk.END).split('\n')
        if len(lines) > 16:
            self.recent_text.delete(1.0, tk.END)
            self.recent_text.insert(1.0, '\n'.join(lines[:15]))
    
    def set_current_status(self, status):
        """設定當前狀態"""
        self.current_label.config(text=status)
    
    def stop(self):
        """停止"""
        self.current_progress.stop()
        self.current_label.config(text="已停止")
    
    def complete(self):
        """完成"""
        self.current_progress.stop()
        self.current_label.config(text="✓ 完成")
    
    def _format_time(self, seconds):
        """格式化時間"""
        if seconds < 60:
            return f"{seconds:.0f} 秒"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m} 分 {s} 秒"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h} 小時 {m} 分"


# ============================================================
# 主GUI（v17）
# ============================================================

class RealtimeAESpectrumPage:
    """即時雙感測器 AE spectrogram 監看頁面。"""

    def __init__(self, parent):
        self.parent = parent
        self.running = False
        self.recording = False
        self.reader = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue(maxsize=8)
        self.ai0_buffer = np.empty(0, dtype=np.float32)
        self.ai1_buffer = np.empty(0, dtype=np.float32)
        self.sample_rate = 1_000_000
        self.baseline = {}
        self.latest_metrics = {}
        self.record_dir = None
        self.record_buffers = []
        self.record_samples = 0
        self.record_index = 0
        self.last_event_time = 0.0
        self.last_plot_time = 0.0
        self.latest_plot_payload = None

        self._build_ui(parent)
        self._schedule_poll()

    def _build_ui(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="即時雙感測器 AE Spectrum 控制", padding=6)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=5)

        self.raw_mode_label = "原始 dB"
        self.delta_mode_label = "相對正常 z-score"
        self.device_var = tk.StringVar(value="cDAQ1Mod1")
        self.ai0_var = tk.StringVar(value="ai0")
        self.ai1_var = tk.StringVar(value="ai1")
        self.sample_rate_var = tk.StringVar(value="1000000")
        self.chunk_var = tk.StringVar(value="50000")
        self.display_sec_var = tk.StringVar(value="60")
        self.simulation_var = tk.BooleanVar(value=not NIDAQ_AVAILABLE)
        self.mode_var = tk.StringVar(value=self.raw_mode_label)
        self.freq_range_var = tk.StringVar(value="40-100 kHz")
        self.smooth_display_var = tk.BooleanVar(value=True)

        entries = [
            ("DAQ裝置", self.device_var, 10),
            ("感測器1", self.ai0_var, 8),
            ("感測器2", self.ai1_var, 8),
            ("取樣率", self.sample_rate_var, 10),
            ("每次讀取點數", self.chunk_var, 8),
            ("累積保留秒數", self.display_sec_var, 6),
        ]
        for idx, (label, var, width) in enumerate(entries):
            ttk.Label(controls, text=label).grid(row=0, column=idx * 2, padx=(3, 1), sticky="w")
            ttk.Entry(controls, textvariable=var, width=width).grid(row=0, column=idx * 2 + 1, padx=(0, 5), sticky="w")

        ttk.Checkbutton(controls, text="模擬模式", variable=self.simulation_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=3, pady=3)
        self.mode_combo = ttk.Combobox(controls, textvariable=self.mode_var, width=10, state="readonly")
        self.mode_combo["values"] = (self.raw_mode_label, self.delta_mode_label)
        self.mode_combo.grid(row=1, column=2, columnspan=2, sticky="w", padx=3)

        self.start_btn = ttk.Button(controls, text="開始", command=self.start)
        self.start_btn.grid(row=1, column=4, padx=3, sticky="ew")
        self.stop_btn = ttk.Button(controls, text="停止", command=self.stop, state="disabled")
        self.stop_btn.grid(row=1, column=5, padx=3, sticky="ew")
        self.baseline_btn = ttk.Button(controls, text="設定正常基準", command=self.set_baseline)
        self.baseline_btn.grid(row=1, column=6, columnspan=2, padx=3, sticky="ew")
        self.record_btn = ttk.Button(controls, text="錄製", command=self.toggle_record)
        self.record_btn.grid(row=1, column=8, padx=3, sticky="ew")
        self.screenshot_btn = ttk.Button(controls, text="截圖", command=self.save_screenshot)
        self.screenshot_btn.grid(row=1, column=9, padx=3, sticky="ew")
        
        # 擷取量程 / 耦合（修量化地板用）
        self.acq_range_var = tk.StringVar(value=ACQ_RANGE_OPTIONS[0][0])
        self.acq_coupling_var = tk.StringVar(value=ACQ_COUPLING_OPTIONS[0])
        self.acq_terminal_var = tk.StringVar(value=ACQ_TERMINAL_OPTIONS[0])
        ttk.Label(controls, text="量程").grid(row=3, column=0, padx=(3, 1), sticky="w")
        ttk.Combobox(controls, textvariable=self.acq_range_var, width=11, state="readonly",
                     values=[lbl for lbl, _ in ACQ_RANGE_OPTIONS]).grid(row=3, column=1, padx=(0, 5), sticky="w")
        ttk.Label(controls, text="耦合").grid(row=3, column=2, padx=(3, 1), sticky="w")
        ttk.Combobox(controls, textvariable=self.acq_coupling_var, width=10, state="readonly",
                     values=ACQ_COUPLING_OPTIONS).grid(row=3, column=3, padx=(0, 5), sticky="w")
        ttk.Label(controls, text="接線").grid(row=3, column=4, padx=(3, 1), sticky="w")
        ttk.Combobox(controls, textvariable=self.acq_terminal_var, width=11, state="readonly",
                     values=ACQ_TERMINAL_OPTIONS).grid(row=3, column=5, padx=(0, 5), sticky="w")

        self.status_var = tk.StringVar(value="就緒。主畫面會分別顯示 ai0 與 ai1，diff 只作為輔助數值。")
        ttk.Label(controls, textvariable=self.status_var, foreground="blue").grid(row=2, column=0, columnspan=12, sticky="w", padx=3, pady=(3, 0))

        plot_frame = ttk.Frame(parent)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.fig = Figure(figsize=(12, 5.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        metrics = ttk.LabelFrame(parent, text="即時數值", padding=5)
        metrics.grid(row=2, column=0, sticky="ew", padx=6, pady=5)
        self.metric_vars = {}
        metric_names = [
            "peak",
            "peak_db",
            "band_35_45",
            "band_100_300",
            "band_250_300",
        ]
        headers = ["通道", "峰值頻率", "峰值dB", "35-45k平均", "100-300k平均", "250-300k平均"]
        for col, text in enumerate(headers):
            ttk.Label(metrics, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=5, sticky="w")
        for row, channel in enumerate(["ai0", "ai1"], start=1):
            ttk.Label(metrics, text=channel, font=("Arial", 9, "bold")).grid(row=row, column=0, padx=5, sticky="w")
            self.metric_vars[channel] = {}
            for col, name in enumerate(metric_names, start=1):
                var = tk.StringVar(value="--")
                self.metric_vars[channel][name] = var
                ttk.Label(metrics, textvariable=var).grid(row=row, column=col, padx=5, sticky="w")

        self.diff_rms_var = tk.StringVar(value="ai1-ai0 RMS: -- | 兩通道峰值差距: --")
        ttk.Label(metrics, textvariable=self.diff_rms_var, foreground="gray").grid(row=3, column=0, columnspan=6, sticky="w", padx=5, pady=(4, 0))
        self._draw_empty()

    def _full_channel_name(self, device, channel):
        channel = str(channel).strip()
        if "/" in channel:
            return channel
        return f"{str(device).strip()}/{channel}"

    def _is_delta_mode(self):
        return self.mode_var.get() == self.delta_mode_label

    def _selected_freq_range(self):
        label = self.freq_range_var.get()
        if label == "40-300 kHz":
            return 40.0, 300.0
        if label == "100-300 kHz":
            return 100.0, 300.0
        return 40.0, 100.0

    def _refresh_frequency_range(self):
        low_khz, high_khz = self._selected_freq_range()
        for ax in self.axes.values():
            ax.set_ylim(low_khz, high_khz)
        self._update_plot(force=True)

    def _parse_config(self):
        fs = int(float(self.sample_rate_var.get()))
        chunk = int(float(self.chunk_var.get()))
        display_sec = float(self.display_sec_var.get())
        if fs < 10_000:
            raise ValueError("sample rate must be >= 10,000 Hz")
        if chunk < 1024:
            raise ValueError("chunk must be >= 1024 samples")
        if display_sec <= 0:
            raise ValueError("display seconds must be positive")
        device = self.device_var.get().strip()
        channels = [
            self._full_channel_name(device, self.ai0_var.get()),
            self._full_channel_name(device, self.ai1_var.get()),
        ]
        return fs, chunk, display_sec, channels

    def start(self):
        if self.running:
            return
        try:
            fs, chunk, _display_sec, channels = self._parse_config()
            simulation = bool(self.simulation_var.get())
            if AEsignal_DualChannelContinuousReader is None and not simulation:
                raise RuntimeError(f"nidaq_capture 連續讀取器無法使用：{NIDAQ_IMPORT_ERROR}")
            self.sample_rate = fs
            self.ai0_buffer = np.empty(0, dtype=np.float32)
            self.ai1_buffer = np.empty(0, dtype=np.float32)
            self.latest_metrics = {}
            self.stop_event.clear()
            acq = parse_acq_controls(self.acq_range_var.get(), self.acq_coupling_var.get(), self.acq_terminal_var.get())
            self.reader = AEsignal_DualChannelContinuousReader(
                samplerate=fs,
                channelnames=channels,
                chunk_size=chunk,
                simulation=simulation,
                **acq,
            ).start()
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            cp_txt = acq["coupling"] or "DC"
            self.status_var.set(
                f"即時監看中：{channels[0]} / {channels[1]}，取樣率 {fs:,} Hz，量程 ±{acq['max_val']}V，{cp_txt} 耦合")
            self.worker_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.worker_thread.start()
        except Exception as exc:
            ErrorHandler.log_error(exc, "start realtime AE spectrum")
            messagebox.showerror("即時 AE Spectrum", str(exc))
            self.status_var.set(f"啟動失敗：{exc}")
            self.stop()

    def stop(self):
        self.stop_event.set()
        self.running = False
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
        self.reader = None
        if self.recording:
            self.toggle_record()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("已停止")

    def _reader_loop(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.reader.read_chunk()
                try:
                    self.data_queue.put(chunk, timeout=0.2)
                except queue.Full:
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.data_queue.put(chunk, timeout=0.2)
            except Exception as exc:
                self.parent.after(0, lambda e=exc: self._reader_failed(e))
                break

    def _reader_failed(self, exc):
        ErrorHandler.log_error(exc, "realtime reader loop")
        self.status_var.set(f"DAQ 串流失敗：{exc}")
        messagebox.showerror("即時 AE Spectrum", str(exc))
        self.stop()

    def _schedule_poll(self):
        self._poll_queue()
        self.parent.after(80, self._schedule_poll)

    def _poll_queue(self):
        got_data = False
        drained = 0
        while drained < 8:
            try:
                chunk = self.data_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1
            ai0 = np.asarray(chunk.get("ai0", []), dtype=np.float32)
            ai1 = np.asarray(chunk.get("ai1", []), dtype=np.float32)
            n = min(ai0.size, ai1.size)
            if n == 0:
                continue
            self.ai0_buffer = np.concatenate([self.ai0_buffer, ai0[:n]])
            self.ai1_buffer = np.concatenate([self.ai1_buffer, ai1[:n]])
            self._trim_buffers()
            if self.recording:
                self._accumulate_record(ai0[:n], ai1[:n])
            got_data = True
        if got_data and time.time() - self.last_plot_time >= 0.8:
            self._update_plot()
            self.last_plot_time = time.time()

    def _trim_buffers(self):
        try:
            display_sec = float(self.display_sec_var.get())
        except Exception:
            display_sec = 2.0
        max_samples = max(4096, int(self.sample_rate * display_sec))
        if self.ai0_buffer.size > max_samples:
            self.ai0_buffer = self.ai0_buffer[-max_samples:]
            self.ai1_buffer = self.ai1_buffer[-max_samples:]

    def _spectrogram_db(self, data):
        x = np.asarray(data, dtype=np.float32)
        if x.size < 2048:
            return None
        x = x - float(np.mean(x))
        nperseg = min(2048, max(512, x.size // 8))
        noverlap = nperseg // 2
        freqs, times, sxx = signal.spectrogram(
            x,
            fs=float(self.sample_rate),
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            mode="psd",
        )
        mask = (freqs >= 40_000) & (freqs <= 300_000)
        if not np.any(mask):
            return None
        db = 10.0 * np.log10(sxx[mask] + 1e-30)
        freqs_khz = freqs[mask] / 1000.0
        max_freq_bins = 260
        max_time_bins = 260
        freq_step = max(1, int(np.ceil(db.shape[0] / max_freq_bins)))
        time_step = max(1, int(np.ceil(db.shape[1] / max_time_bins)))
        return freqs_khz[::freq_step], times[::time_step], db[::freq_step, ::time_step]

    def _band_mean(self, freqs_khz, db, low, high):
        mask = (freqs_khz >= low) & (freqs_khz <= high)
        if not np.any(mask):
            return float("nan")
        return float(np.nanmean(db[mask, :]))

    def _metrics(self, freqs_khz, times, db):
        idx = int(np.nanargmax(db))
        freq_idx, time_idx = np.unravel_index(idx, db.shape)
        return {
            "peak_freq_khz": float(freqs_khz[freq_idx]),
            "peak_time_sec": float(times[time_idx]),
            "peak_db": float(db[freq_idx, time_idx]),
            "band_35_45_db": self._band_mean(freqs_khz, db, 35, 45),
            "band_40_100_db": self._band_mean(freqs_khz, db, 40, 100),
            "band_100_300_db": self._band_mean(freqs_khz, db, 100, 300),
            "band_250_300_db": self._band_mean(freqs_khz, db, 250, 300),
            "band_40_100_max_db": float(np.nanmax(db[(freqs_khz >= 40) & (freqs_khz <= 100), :])) if np.any((freqs_khz >= 40) & (freqs_khz <= 100)) else float("nan"),
        }

    def _plot_channel(self, ax, channel, spec, baseline):
        freqs_khz, times, raw_db = spec
        plot_db = raw_db
        color_label = "dB"
        if self._is_delta_mode() and baseline is not None:
            rows = min(raw_db.shape[0], baseline.shape[0])
            cols = min(raw_db.shape[1], baseline.shape[1])
            freqs_khz = freqs_khz[:rows]
            times = times[:cols]
            plot_db = raw_db[:rows, :cols] - baseline[:rows, :cols]
            color_label = "delta dB"
        vmin, vmax = np.nanpercentile(plot_db, [2, 99])
        if self._is_delta_mode():
            limit = max(abs(vmin), abs(vmax), 1.0)
            vmin, vmax = -limit, limit
            cmap = "coolwarm"
        else:
            cmap = "magma"
        im = ax.pcolormesh(times, freqs_khz, plot_db, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axhline(40, color="black" if self._is_delta_mode() else "white", linestyle="--", linewidth=1.1)
        ax.axhspan(100, 300, color="white", alpha=0.08)
        metrics = self._metrics(freqs_khz, times, plot_db)
        ax.scatter([metrics["peak_time_sec"]], [metrics["peak_freq_khz"]], s=42, facecolors="none", edgecolors="cyan", linewidths=1.4)
        ax.set_title(f"{channel} 感測器 Spectrum")
        ax.set_xlabel("時間 (s)")
        ax.set_ylabel("頻率 (kHz)")
        ax.set_ylim(40, 300)
        ax.grid(alpha=0.15)
        return im, color_label, metrics

    def _update_plot(self):
        ai0_spec = self._spectrogram_db(self.ai0_buffer)
        ai1_spec = self._spectrogram_db(self.ai1_buffer)
        if ai0_spec is None or ai1_spec is None:
            return
        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        im0, label0, m0 = self._plot_channel(axes[0], "ai0", ai0_spec, self.baseline.get("ai0"))
        im1, label1, m1 = self._plot_channel(axes[1], "ai1", ai1_spec, self.baseline.get("ai1"))
        self.fig.colorbar(im0, ax=axes[0], label=label0)
        self.fig.colorbar(im1, ax=axes[1], label=label1)
        self.fig.suptitle("即時雙感測器 AE Spectrogram（ai0 / ai1 分開顯示）", y=0.98)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.latest_metrics = {"ai0": m0, "ai1": m1}
        self.latest_plot_payload = {"ai0": ai0_spec, "ai1": ai1_spec}
        self._update_metric_labels()
        self._maybe_save_event()

    def _draw_empty(self):
        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        for ax, title in zip(axes, ["ai0 感測器 Spectrum", "ai1 感測器 Spectrum"]):
            ax.set_title(title)
            ax.set_xlabel("時間 (s)")
            ax.set_ylabel("頻率 (kHz)")
            ax.set_ylim(40, 300)
            ax.axhline(40, color="black", linestyle="--", linewidth=1.0)
            ax.axhspan(100, 300, color="#dddddd", alpha=0.25)
            ax.text(0.5, 0.5, "按「開始」", ha="center", va="center", transform=ax.transAxes)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _update_metric_labels(self):
        for channel in ["ai0", "ai1"]:
            metrics = self.latest_metrics.get(channel, {})
            if not metrics:
                continue
            self.metric_vars[channel]["peak"].set(f"{metrics['peak_freq_khz']:.1f} kHz")
            self.metric_vars[channel]["peak_db"].set(f"{metrics['peak_db']:.1f} dB")
            self.metric_vars[channel]["band_35_45"].set(f"{metrics['band_35_45_db']:.1f} dB")
            self.metric_vars[channel]["band_100_300"].set(f"{metrics['band_100_300_db']:.1f} dB")
            self.metric_vars[channel]["band_250_300"].set(f"{metrics['band_250_300_db']:.1f} dB")
        n = min(self.ai0_buffer.size, self.ai1_buffer.size)
        if n:
            diff = self.ai1_buffer[-n:] - self.ai0_buffer[-n:]
            diff_rms = float(np.sqrt(np.mean(diff * diff)))
        else:
            diff_rms = float("nan")
        ai0_peak = self.latest_metrics.get("ai0", {}).get("peak_freq_khz")
        ai1_peak = self.latest_metrics.get("ai1", {}).get("peak_freq_khz")
        if ai0_peak is not None and ai1_peak is not None:
            agreement = f"相差 {abs(ai1_peak - ai0_peak):.1f} kHz"
        else:
            agreement = "--"
        self.diff_rms_var.set(f"ai1-ai0 RMS: {diff_rms:.4g} | 兩通道峰值差距: {agreement}")

    def set_baseline(self):
        ai0_spec = self._spectrogram_db(self.ai0_buffer)
        ai1_spec = self._spectrogram_db(self.ai1_buffer)
        if ai0_spec is None or ai1_spec is None:
            messagebox.showwarning("正常基準", "需要更多即時資料才能設定正常基準。")
            return
        self.baseline["ai0"] = ai0_spec[2].copy()
        self.baseline["ai1"] = ai1_spec[2].copy()
        self.mode_var.set(self.delta_mode_label)
        self.status_var.set("已儲存 ai0 / ai1 正常基準，畫面切換為相對正常 Delta dB。")

    def _session_dir(self):
        if self.record_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_dir = PROGRAM_DIR / "realtime_spectrum_events" / f"{stamp}_session"
            self.record_dir.mkdir(parents=True, exist_ok=True)
        return self.record_dir

    def toggle_record(self):
        if not self.recording:
            self._session_dir()

            self.recording = True
            self.record_buffers = []
            self.record_samples = 0
            self.record_index = 0
            self.record_btn.config(text="停止錄製")
            self.status_var.set(f"正在錄製原始 ai0/ai1 到 {self.record_dir}")
        else:
            self._flush_record(force=True)
            self.recording = False
            self.record_btn.config(text="錄製")
            self.status_var.set(f"錄製停止：{self.record_dir}")

    def _accumulate_record(self, ai0, ai1):
        self.record_buffers.append((np.asarray(ai0, dtype=np.float32), np.asarray(ai1, dtype=np.float32)))
        self.record_samples += min(ai0.size, ai1.size)
        if self.record_samples >= self.sample_rate:
            self._flush_record(force=False)

    def _flush_record(self, force=False):
        if not self.record_buffers or (not force and self.record_samples < self.sample_rate):
            return
        ai0 = np.concatenate([pair[0] for pair in self.record_buffers])
        ai1 = np.concatenate([pair[1] for pair in self.record_buffers])
        n = min(ai0.size, ai1.size)
        path = self._session_dir() / f"record_{self.record_index:04d}.npz"
        np.savez_compressed(
            path,
            ai0=ai0[:n].astype(np.float32),
            ai1=ai1[:n].astype(np.float32),
            sample_rate=np.asarray([self.sample_rate], dtype=np.float64),
            created_at=np.asarray([datetime.now().isoformat()]),
        )
        self.record_index += 1
        self.record_buffers = []
        self.record_samples = 0
        

    def save_screenshot(self):
        out_dir = self._session_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = out_dir / f"spectrum_{stamp}.png"
        csv_path = out_dir / f"spectrum_metrics_{stamp}.csv"
        self.fig.savefig(png_path, dpi=160)
        self._write_metrics_csv(csv_path)
        self.status_var.set(f"已儲存截圖：{png_path}")

    def _write_metrics_csv(self, path):
        rows = []
        for channel, metrics in self.latest_metrics.items():
            row = {"timestamp": datetime.now().isoformat(), "channel": channel, "mode": self.mode_var.get()}
            row.update(metrics)
            rows.append(row)
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _maybe_save_event(self):
        now = time.time()
        if now - self.last_event_time < 5.0:
            return
        if not self._is_delta_mode():
            return
        triggered = False
        for metrics in self.latest_metrics.values():
            if metrics.get("peak_db", -999) >= 10.0 or metrics.get("band_250_300_db", -999) >= 1.0:
                triggered = True
        if not triggered:
            return
        self.last_event_time = now
        out_dir = self._session_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = out_dir / f"event_spectrum_{stamp}.png"
        csv_path = out_dir / f"event_metrics_{stamp}.csv"
        self.fig.savefig(png_path, dpi=170)
        self._write_metrics_csv(csv_path)
        self.status_var.set(f"已儲存事件截圖：{png_path}")


class StableRealtimeAESpectrumPage:
    """穩定版即時雙感測器 AE spectrogram 監看頁面。"""

    def __init__(self, parent):
        self.parent = parent
        self.running = False
        self.recording = False
        self.reader = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.suppress_reader_errors = False
        self.error_popup_shown = False

        self.sample_rate = 1_000_000
        self.chunk_size = 50_000
        self.ring_lock = threading.Lock()
        self.ring_capacity = 1
        self.ring_ai0 = np.zeros(1, dtype=np.float32)
        self.ring_ai1 = np.zeros(1, dtype=np.float32)
        self.ring_write = 0
        self.ring_count = 0
        self.ai0_buffer = np.empty(0, dtype=np.float32)
        self.ai1_buffer = np.empty(0, dtype=np.float32)

        self.baseline = {}
        self.dual_baseline = {}
        self.latest_metrics = {}
        self.latest_dual_metrics = {}
        self.latest_plot_payload = None
        self.last_event_time = 0.0
        self.last_plot_time = 0.0
        self.baseline_seconds = 5.0
        self.baseline_min_seconds = 3.0
        self.delta_color_limit_db = 1.5
        # 升級後 Delta 模式以 robust z-score(σ) 呈現，門檻改為標準差倍數
        self.delta_warn_db = 3.0
        self.delta_alert_db = 5.0
        self.visual_cache_seconds = 60.0

        # 去噪 / 變異縮減設定（基底 EMA 永遠開，其餘可開關+調參）
        self.ema_alpha = 0.3              # 時間 EMA 係數，越小越平滑
        self.z_color_limit = 6.0          # z-score 顯示色階上限 (±σ)
        self.z_mad_floor_db = 0.5         # MAD 下限，避免安靜頻段 z 爆衝
        self.coh_gate_enabled = False
        self.coh_gate_threshold = 0.3
        self.spec_sub_enabled = False
        self.spec_sub_alpha = 1.5         # over-subtraction 係數
        self.spec_sub_floor = 0.1         # 殘留地板係數 β
        self.wavelet_enabled = False
        self.wavelet_name = "db6"
        self.wavelet_level = 4
        self.wavelet_k = 1.0
        self._last_coh_profile = None
        self.total_samples_read = 0

        # 洩漏偵測（1-40kHz band-RMS + 4 detectors；按「校準偵測器」啟用）
        self.detect_band_lo = 1000.0
        self.detect_band_hi = 40_000.0
        self.detect_win_sec = 0.1
        self.detect_enabled = False
        self.detect_band_sos = None
        self.detect_win_n = 0
        self.detect_acc0 = np.empty(0, dtype=np.float64)
        self.detect_acc1 = np.empty(0, dtype=np.float64)
        self.detect_adaptive = None
        self.detect_cusum = None
        self.detect_burst = None
        self.detect_refsub_obj = None
        self.detect_refsub_adaptive = None
        self.detect_latest = None      # {rms, cusum_S, burst_rate, residual, triggers, adaptive_thr}
        self.detect_lock = threading.Lock()
        self.plot_lock = threading.Lock()
        self.plot_pending_ai0 = []
        self.plot_pending_ai1 = []
        self.plot_pending_start_sample = 0
        self.plot_pending_samples = 0
        self.plot_cache = {}

        self.record_lock = threading.Lock()
        self.record_dir = None
        self.record_buffers = []
        self.record_samples = 0
        self.record_index = 0

        self.axes = {}
        self.images = {}
        self.colorbars = {}
        self.peak_markers = {}
        self.trend_axes = {}
        self.trend_lines = {}
        self.trend_hint_text = {}
        self.trend_times = []
        self.trend_values = {}
        self.dual_trend_ax = None
        self.dual_trend_lines = {}
        self.dual_trend_hint_text = None
        self.dual_trend_values = {}
        
        self._build_ui(parent)
        self._schedule_poll()

    def _build_ui(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=3)
        parent.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(parent, text="即時雙感測器 AE Spectrum 控制", padding=6)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=5)

        self.raw_mode_label = "原始 dB"
        self.delta_mode_label = "相對正常 z-score"
        self.device_var = tk.StringVar(value="cDAQ1Mod1")
        self.ai0_var = tk.StringVar(value="ai0")
        self.ai1_var = tk.StringVar(value="ai1")
        self.sample_rate_var = tk.StringVar(value="1000000")
        self.chunk_var = tk.StringVar(value="50000")
        self.display_sec_var = tk.StringVar(value="60")
        self.simulation_var = tk.BooleanVar(value=not NIDAQ_AVAILABLE)
        self.mode_var = tk.StringVar(value=self.raw_mode_label)
        self.freq_range_var = tk.StringVar(value="40-100 kHz")
        self.smooth_display_var = tk.BooleanVar(value=True)

        # 去噪設定面板用變數
        self.ema_alpha_var = tk.StringVar(value="0.3")
        self.z_limit_var = tk.StringVar(value="6.0")
        self.coh_gate_var = tk.BooleanVar(value=False)
        self.coh_thr_var = tk.StringVar(value="0.30")
        self.spec_sub_var = tk.BooleanVar(value=False)
        self.spec_alpha_var = tk.StringVar(value="1.5")
        self.spec_floor_var = tk.StringVar(value="0.1")
        self.wavelet_var = tk.BooleanVar(value=False)
        self.wavelet_name_var = tk.StringVar(value="db6")
        self.wavelet_level_var = tk.StringVar(value="4")
        self.wavelet_k_var = tk.StringVar(value="1.0")

        entries = [
            ("DAQ 裝置", self.device_var, 10),
            ("感測器 1", self.ai0_var, 8),
            ("感測器 2", self.ai1_var, 8),
            ("取樣率", self.sample_rate_var, 10),
            ("每次讀取點數", self.chunk_var, 8),
            ("累積保留秒數", self.display_sec_var, 6),
        ]
        for idx, (label, var, width) in enumerate(entries):
            ttk.Label(controls, text=label).grid(row=0, column=idx * 2, padx=(3, 1), sticky="w")
            ttk.Entry(controls, textvariable=var, width=width).grid(row=0, column=idx * 2 + 1, padx=(0, 5), sticky="w")

        ttk.Checkbutton(controls, text="模擬模式", variable=self.simulation_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=3, pady=3)
        self.mode_combo = ttk.Combobox(controls, textvariable=self.mode_var, width=16, state="readonly")
        self.mode_combo["values"] = (self.raw_mode_label, self.delta_mode_label)
        self.mode_combo.grid(row=1, column=2, columnspan=2, sticky="w", padx=3)

        ttk.Label(controls, text="頻率範圍").grid(row=1, column=10, padx=(8, 1), sticky="e")
        self.freq_range_combo = ttk.Combobox(controls, textvariable=self.freq_range_var, width=12, state="readonly")
        self.freq_range_combo["values"] = ("40-100 kHz", "40-300 kHz", "100-300 kHz")
        self.freq_range_combo.grid(row=1, column=11, sticky="w", padx=3)
        self.freq_range_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_frequency_range())
        ttk.Checkbutton(controls, text="平滑顯示", variable=self.smooth_display_var).grid(row=1, column=12, columnspan=2, sticky="w", padx=3)

        self.start_btn = ttk.Button(controls, text="開始", command=self.start)
        self.start_btn.grid(row=1, column=4, padx=3, sticky="ew")
        self.stop_btn = ttk.Button(controls, text="停止", command=self.stop, state="disabled")
        self.stop_btn.grid(row=1, column=5, padx=3, sticky="ew")
        self.baseline_btn = ttk.Button(controls, text="設定正常基準", command=self.set_baseline)
        self.baseline_btn.grid(row=1, column=6, columnspan=2, padx=3, sticky="ew")
        self.record_btn = ttk.Button(controls, text="錄製", command=self.toggle_record)
        self.record_btn.grid(row=1, column=8, padx=3, sticky="ew")
        self.screenshot_btn = ttk.Button(controls, text="截圖", command=self.save_screenshot)
        self.screenshot_btn.grid(row=1, column=9, padx=3, sticky="ew")

        # 擷取量程 / 耦合（修量化地板用）
        self.acq_range_var = tk.StringVar(value=ACQ_RANGE_OPTIONS[0][0])
        self.acq_coupling_var = tk.StringVar(value=ACQ_COUPLING_OPTIONS[0])
        self.acq_terminal_var = tk.StringVar(value=ACQ_TERMINAL_OPTIONS[0])
        ttk.Label(controls, text="量程").grid(row=4, column=0, padx=(3, 1), sticky="w")
        ttk.Combobox(controls, textvariable=self.acq_range_var, width=11, state="readonly",
                     values=[lbl for lbl, _ in ACQ_RANGE_OPTIONS]).grid(row=4, column=1, padx=(0, 5), sticky="w")
        ttk.Label(controls, text="耦合").grid(row=4, column=2, padx=(3, 1), sticky="w")
        ttk.Combobox(controls, textvariable=self.acq_coupling_var, width=10, state="readonly",
                     values=ACQ_COUPLING_OPTIONS).grid(row=4, column=3, padx=(0, 5), sticky="w")
        ttk.Label(controls, text="接線").grid(row=4, column=4, padx=(3, 1), sticky="w")
        ttk.Combobox(controls, textvariable=self.acq_terminal_var, width=11, state="readonly",
                     values=ACQ_TERMINAL_OPTIONS).grid(row=4, column=5, padx=(0, 5), sticky="w")

        # 洩漏偵測 UI（1-40kHz band-RMS + 4 個 detector）
        self.detect_status_var = tk.StringVar(value="未校準")
        self.detect_rms_var = tk.StringVar(value="--")
        self.detect_cusum_var = tk.StringVar(value="--")
        self.detect_burst_var = tk.StringVar(value="--")
        self.detect_res_var = tk.StringVar(value="--")
        detect_frame = ttk.LabelFrame(controls, text="洩漏偵測 (1-40kHz band-RMS)", padding=4)
        detect_frame.grid(row=5, column=0, columnspan=14, sticky="ew", padx=3, pady=(5, 0))
        ttk.Label(detect_frame, text="狀態:").grid(row=0, column=0, padx=(3, 1), sticky="w")
        self.detect_status_label = ttk.Label(detect_frame, textvariable=self.detect_status_var,
                                              foreground="gray", font=("Arial", 10, "bold"))
        self.detect_status_label.grid(row=0, column=1, padx=(0, 14), sticky="w")
        ttk.Label(detect_frame, text="RMS:").grid(row=0, column=2, padx=(0, 1))
        ttk.Label(detect_frame, textvariable=self.detect_rms_var, width=20).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(detect_frame, text="CUSUM S:").grid(row=0, column=4, padx=(0, 1))
        ttk.Label(detect_frame, textvariable=self.detect_cusum_var, width=14).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(detect_frame, text="Burst率:").grid(row=0, column=6, padx=(0, 1))
        ttk.Label(detect_frame, textvariable=self.detect_burst_var, width=8).grid(row=0, column=7, padx=(0, 8))
        ttk.Label(detect_frame, text="RefSub殘差:").grid(row=0, column=8, padx=(0, 1))
        ttk.Label(detect_frame, textvariable=self.detect_res_var, width=12).grid(row=0, column=9, padx=(0, 8))
        ttk.Button(detect_frame, text="校準偵測器", command=self._calibrate_detectors).grid(row=0, column=10, padx=4)
        ttk.Button(detect_frame, text="重置 CUSUM", command=self._reset_cusum).grid(row=0, column=11, padx=4)

        self.status_var = tk.StringVar(value="就緒。ai0 / ai1 分別顯示，DAQ 讀取與 GUI 繪圖已分離。")
        ttk.Label(controls, textvariable=self.status_var, foreground="blue").grid(row=2, column=0, columnspan=14, sticky="w", padx=3, pady=(3, 0))

        denoise = ttk.LabelFrame(controls, text="去噪設定（變異縮減為基底，永遠開啟）", padding=4)
        denoise.grid(row=3, column=0, columnspan=14, sticky="ew", padx=3, pady=(5, 0))
        ttk.Label(denoise, text="EMA α").grid(row=0, column=0, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.ema_alpha_var, width=5).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(denoise, text="z 色階±σ").grid(row=0, column=2, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.z_limit_var, width=5).grid(row=0, column=3, padx=(0, 12))
        ttk.Checkbutton(denoise, text="相干遮罩", variable=self.coh_gate_var).grid(row=0, column=4, sticky="w")
        ttk.Label(denoise, text="門檻").grid(row=0, column=5, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.coh_thr_var, width=5).grid(row=0, column=6, padx=(0, 12))
        ttk.Checkbutton(denoise, text="譜減法", variable=self.spec_sub_var).grid(row=0, column=7, sticky="w")
        ttk.Label(denoise, text="α").grid(row=0, column=8, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.spec_alpha_var, width=5).grid(row=0, column=9, padx=(0, 4))
        ttk.Label(denoise, text="地板β").grid(row=0, column=10, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.spec_floor_var, width=5).grid(row=0, column=11, padx=(0, 12))
        wavelet_text = "小波包去噪" if PYWT_AVAILABLE else "小波包去噪(需 pywt)"
        self.wavelet_check = ttk.Checkbutton(denoise, text=wavelet_text, variable=self.wavelet_var)
        self.wavelet_check.grid(row=0, column=12, sticky="w")
        if not PYWT_AVAILABLE:
            self.wavelet_check.state(["disabled"])
        ttk.Label(denoise, text="wavelet").grid(row=0, column=13, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.wavelet_name_var, width=6).grid(row=0, column=14, padx=(0, 4))
        ttk.Label(denoise, text="level").grid(row=0, column=15, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.wavelet_level_var, width=4).grid(row=0, column=16, padx=(0, 4))
        ttk.Label(denoise, text="k").grid(row=0, column=17, padx=(3, 1), sticky="e")
        ttk.Entry(denoise, textvariable=self.wavelet_k_var, width=4).grid(row=0, column=18, padx=(0, 4))

        plot_frame = ttk.Frame(parent)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.fig = Figure(figsize=(12, 5.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._init_plot()

        trend_frame = ttk.LabelFrame(parent, text="頻帶 Delta 趨勢圖", padding=4)
        trend_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 4))
        trend_frame.columnconfigure(0, weight=1)
        trend_frame.rowconfigure(0, weight=1)
        self.trend_fig = Figure(figsize=(12, 2.25), dpi=100)
        self.trend_canvas = FigureCanvasTkAgg(self.trend_fig, trend_frame)
        self.trend_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._init_trend_plot()

        metrics = ttk.LabelFrame(parent, text="微洩漏判讀指標（相對正常 z-score σ）", padding=5)
        metrics.grid(row=3, column=0, sticky="ew", padx=6, pady=5)
        self.metric_vars = {}
        self.metric_widgets = {}
        metric_names = ["peak", "peak_db", "band_35_45", "band_40_100", "band_40_100_max", "band_100_300", "band_250_300", "status"]
        headers = ["通道", "最大 z 頻率", "最大 z(σ)", "35-45k zΔ", "40-100k 平均z", "40-100k 最大z", "100-300k zΔ", "250-300k zΔ", "狀態"]
        for col, text in enumerate(headers):
            ttk.Label(metrics, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=5, sticky="w")
        for row, channel in enumerate(["ai0", "ai1"], start=1):
            ttk.Label(metrics, text=channel, font=("Arial", 9, "bold")).grid(row=row, column=0, padx=5, sticky="w")
            self.metric_vars[channel] = {}
            self.metric_widgets[channel] = {}
            for col, name in enumerate(metric_names, start=1):
                var = tk.StringVar(value="--")
                self.metric_vars[channel][name] = var
                label = ttk.Label(metrics, textvariable=var)
                label.grid(row=row, column=col, padx=5, sticky="w")
                self.metric_widgets[channel][name] = label

        self.diff_rms_var = tk.StringVar(value="ai1-ai0 RMS: -- | 100-300k 通道差: -- | 250-300k 通道差: --")
        ttk.Label(metrics, textvariable=self.diff_rms_var, foreground="gray").grid(row=3, column=0, columnspan=9, sticky="w", padx=5, pady=(4, 0))
        self.baseline_hint_var = tk.StringVar(value="提示：先在正常狀態收 3-5 秒後按「設定正常基準」，再用 z-score 判讀微洩漏（背景 ~±1σ，真實窄帶 z≫3）。")
        ttk.Label(metrics, textvariable=self.baseline_hint_var, foreground="gray").grid(row=4, column=0, columnspan=9, sticky="w", padx=5, pady=(2, 0))
        ttk.Separator(metrics, orient="horizontal").grid(row=5, column=0, columnspan=9, sticky="ew", padx=5, pady=(6, 3))
        dual_headers = ["雙通道", "35-45k cohΔ", "40-100k cohΔ", "100-300k cohΔ", "250-300k cohΔ", "40-100k CPSDΔ", "GCC 峰值比", "延遲(ms)", "狀態"]
        self.dual_metric_vars = {}
        self.dual_metric_widgets = {}
        for col, text in enumerate(dual_headers):
            ttk.Label(metrics, text=text, font=("Arial", 9, "bold")).grid(row=6, column=col, padx=5, sticky="w")
        ttk.Label(metrics, text="ai0↔ai1", font=("Arial", 9, "bold")).grid(row=7, column=0, padx=5, sticky="w")
        for col, name in enumerate(["coh_35_45", "coh_40_100", "coh_100_300", "coh_250_300", "cpsd_40_100", "gcc_score", "delay_ms", "status"], start=1):
            var = tk.StringVar(value="--")
            self.dual_metric_vars[name] = var
            label = ttk.Label(metrics, textvariable=var)
            label.grid(row=7, column=col, padx=5, sticky="w")
            self.dual_metric_widgets[name] = label

    def _init_plot(self):
        self.fig.clear()
        axes = self.fig.subplots(1, 2)
        blank = np.zeros((2, 2), dtype=np.float32)
        low_khz, high_khz = self._selected_freq_range()
        for ax, channel, title in zip(axes, ["ai0", "ai1"], ["ai0 感測器 Spectrum", "ai1 感測器 Spectrum"]):
            im = ax.imshow(
                blank,
                origin="lower",
                aspect="auto",
                extent=[0, 1.0, low_khz, high_khz],
                cmap="magma",
                vmin=-120,
                vmax=0,
            )
            ax.axhline(40, color="white", linestyle="--", linewidth=1.0)
            ax.axhline(100, color="white", linestyle=":", linewidth=0.9, alpha=0.8)
            if high_khz > 100:
                ax.axhspan(100, high_khz, color="white", alpha=0.08)
            marker = ax.scatter([], [], s=58, facecolors="none", edgecolors="cyan", linewidths=1.6)
            ax.set_title(title)
            ax.set_xlabel("累積時間 (s)")
            ax.set_ylabel("頻率 (kHz)")
            ax.set_ylim(low_khz, high_khz)
            ax.grid(alpha=0.15)
            cb = self.fig.colorbar(im, ax=ax, label="dB（Raw 訊號品質檢查）")
            self.axes[channel] = ax
            self.images[channel] = im
            self.colorbars[channel] = cb
            self.peak_markers[channel] = marker
        self.fig.suptitle("即時雙感測器 AE Spectrogram（ai0 / ai1 分開顯示）", y=0.98)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _trend_band_defs(self):
        return [
            ("delta_band_40_100_db", "40-100k z", "#c0392b", 2.4, "-"),
            ("delta_band_35_45_db", "35-45k z", "#2874a6", 1.4, "--"),
            ("delta_band_100_300_db", "100-300k z", "#1e8449", 1.4, "-."),
            ("delta_band_250_300_db", "250-300k z", "#7d3c98", 1.4, ":"),
        ]

    def _dual_trend_defs(self):
        return [
            ("coh_delta_40_100", "40-100k cohΔ", "#c0392b", 2.4, "-"),
            ("coh_delta_35_45", "35-45k cohΔ", "#2874a6", 1.4, "--"),
            ("coh_delta_100_300", "100-300k cohΔ", "#1e8449", 1.4, "-."),
            ("coh_delta_250_300", "250-300k cohΔ", "#7d3c98", 1.4, ":"),
        ]

    def _reset_trend_cache(self):
        self.trend_times = []
        self.trend_values = {
            channel: {key: [] for key, _label, _color, _width, _style in self._trend_band_defs()}
            for channel in ["ai0", "ai1"]
        }
        self.dual_trend_values = {key: [] for key, _label, _color, _width, _style in self._dual_trend_defs()}

    def _init_trend_plot(self):
        self.trend_fig.clear()
        axes = self.trend_fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]})
        self.trend_axes = {}
        self.trend_lines = {}
        self.trend_hint_text = {}
        self.dual_trend_ax = None
        self.dual_trend_lines = {}
        self.dual_trend_hint_text = None
        self._reset_trend_cache()

        for ax, channel in zip(axes[:2], ["ai0", "ai1"]):
            self.trend_axes[channel] = ax
            self.trend_lines[channel] = {}
            ax.axhline(0.0, color="#555555", linewidth=0.9, alpha=0.8)
            ax.axhline(self.delta_warn_db, color="#d35400", linewidth=0.8, linestyle="--", alpha=0.65)
            ax.axhline(self.delta_alert_db, color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.75)
            ax.axhline(-self.delta_warn_db, color="#21618c", linewidth=0.8, linestyle=":", alpha=0.5)
            for key, label, color, width, style in self._trend_band_defs():
                line, = ax.plot([], [], linestyle=style, color=color, linewidth=width, label=label)
                self.trend_lines[channel][key] = line
            self.trend_hint_text[channel] = ax.text(
                0.5,
                0.52,
                "設定正常基準後顯示頻帶趨勢",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
            )
            ax.set_title(f"{channel} 頻帶 z 趨勢")
            ax.set_xlabel("累積時間 (s)")
            ax.set_ylim(-self.z_color_limit, self.z_color_limit)
            ax.grid(alpha=0.18)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        axes[0].set_ylabel("相對正常 z (σ)")

        dual_ax = axes[2]
        self.dual_trend_ax = dual_ax
        dual_ax.axhline(0.0, color="#555555", linewidth=0.9, alpha=0.8)
        dual_ax.axhline(0.15, color="#d35400", linewidth=0.8, linestyle="--", alpha=0.65)
        dual_ax.axhline(0.30, color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.75)
        dual_ax.axhline(-0.15, color="#21618c", linewidth=0.8, linestyle=":", alpha=0.5)
        for key, label, color, width, style in self._dual_trend_defs():
            line, = dual_ax.plot([], [], linestyle=style, color=color, linewidth=width, label=label)
            self.dual_trend_lines[key] = line
        self.dual_trend_hint_text = dual_ax.text(
            0.5,
            0.52,
            "設定正常基準後顯示 coherence Δ",
            transform=dual_ax.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=10,
        )
        dual_ax.set_title("雙通道相干趨勢")
        dual_ax.set_xlabel("累積時間 (s)")
        dual_ax.set_ylabel("coherence Δ")
        dual_ax.set_ylim(-0.35, 0.55)
        dual_ax.grid(alpha=0.18)
        dual_ax.legend(loc="upper right", fontsize=8, frameon=False)
        self.trend_fig.tight_layout()
        self.trend_canvas.draw_idle()

    def _clear_trend(self, redraw=True):
        self._reset_trend_cache()
        for channel_lines in self.trend_lines.values():
            for line in channel_lines.values():
                line.set_data([], [])
        for text in self.trend_hint_text.values():
            text.set_visible(True)
        for line in self.dual_trend_lines.values():
            line.set_data([], [])
        if self.dual_trend_hint_text is not None:
            self.dual_trend_hint_text.set_visible(True)
        if redraw and hasattr(self, "trend_canvas"):
            self._update_trend_plot()

    def _append_trend_point(self, elapsed_sec):
        if not self.latest_metrics:
            return False
        has_delta = any(self.latest_metrics.get(channel, {}).get("has_delta") for channel in ["ai0", "ai1"])
        if not has_delta:
            return False
        if not self.trend_values:
            self._reset_trend_cache()

        self.trend_times.append(float(elapsed_sec))
        for channel in ["ai0", "ai1"]:
            metrics = self.latest_metrics.get(channel, {})
            for key, _label, _color, _width, _style in self._trend_band_defs():
                value = metrics.get(key, float("nan"))
                self.trend_values[channel][key].append(float(value) if np.isfinite(value) else float("nan"))
        dual_metrics = self.latest_dual_metrics or {}
        for key, _label, _color, _width, _style in self._dual_trend_defs():
            value = dual_metrics.get(key, float("nan"))
            self.dual_trend_values[key].append(float(value) if np.isfinite(value) else float("nan"))
        self._prune_trend_cache()
        return True

    def _prune_trend_cache(self):
        if not self.trend_times:
            return
        max_time = float(self.trend_times[-1])
        cutoff = max(0.0, max_time - float(self.visual_cache_seconds))
        keep = [idx for idx, value in enumerate(self.trend_times) if value >= cutoff]
        if len(keep) == len(self.trend_times):
            return
        self.trend_times = [self.trend_times[idx] for idx in keep]
        for channel in ["ai0", "ai1"]:
            for key in self.trend_values.get(channel, {}):
                values = self.trend_values[channel][key]
                self.trend_values[channel][key] = [values[idx] for idx in keep if idx < len(values)]
        for key, values in self.dual_trend_values.items():
            self.dual_trend_values[key] = [values[idx] for idx in keep if idx < len(values)]

    def _update_trend_plot(self, elapsed_sec=None):
        times = np.asarray(self.trend_times, dtype=float)
        finite_values = []
        finite_dual_values = []
        for channel in ["ai0", "ai1"]:
            for key, _label, _color, _width, _style in self._trend_band_defs():
                values = np.asarray(self.trend_values.get(channel, {}).get(key, []), dtype=float)
                line = self.trend_lines.get(channel, {}).get(key)
                if line is not None:
                    line.set_data(times[:values.size], values)
                finite_values.extend(values[np.isfinite(values)].tolist())
        for key, _label, _color, _width, _style in self._dual_trend_defs():
            values = np.asarray(self.dual_trend_values.get(key, []), dtype=float)
            line = self.dual_trend_lines.get(key)
            if line is not None:
                line.set_data(times[:values.size], values)
            finite_dual_values.extend(values[np.isfinite(values)].tolist())

        has_points = bool(finite_values)
        if elapsed_sec is None:
            elapsed_sec = float(times[-1]) if times.size else 0.0
        right = max(float(elapsed_sec), float(times[-1]) if times.size else 0.0, 1.0)
        left = max(0.0, right - float(self.visual_cache_seconds))

        if finite_values:
            low = min(-self.delta_color_limit_db, float(np.nanpercentile(finite_values, 2)) - 0.2)
            high = max(self.delta_color_limit_db, float(np.nanpercentile(finite_values, 98)) + 0.2)
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = -self.delta_color_limit_db, self.delta_color_limit_db
        else:
            low, high = -self.delta_color_limit_db, self.delta_color_limit_db

        for ax in self.trend_axes.values():
            ax.set_xlim(left, right)
            ax.set_ylim(low, high)
        for text in self.trend_hint_text.values():
            text.set_visible(not has_points)
        if self.dual_trend_ax is not None:
            self.dual_trend_ax.set_xlim(left, right)
            if finite_dual_values:
                dual_low = min(-0.35, float(np.nanpercentile(finite_dual_values, 2)) - 0.05)
                dual_high = max(0.55, float(np.nanpercentile(finite_dual_values, 98)) + 0.05)
                if not np.isfinite(dual_low) or not np.isfinite(dual_high) or dual_low == dual_high:
                    dual_low, dual_high = -0.35, 0.55
            else:
                dual_low, dual_high = -0.35, 0.55
            self.dual_trend_ax.set_ylim(dual_low, dual_high)
        if self.dual_trend_hint_text is not None:
            self.dual_trend_hint_text.set_visible(not bool(finite_dual_values))
        if hasattr(self, "trend_canvas"):
            self.trend_canvas.draw_idle()

    def _full_channel_name(self, device, channel):
        channel = str(channel).strip()
        if "/" in channel:
            return channel
        return f"{str(device).strip()}/{channel}"

    def _is_delta_mode(self):
        return self.mode_var.get() == self.delta_mode_label

    def _selected_freq_range(self):
        label = self.freq_range_var.get()
        if label == "40-300 kHz":
            return 40.0, 300.0
        if label == "100-300 kHz":
            return 100.0, 300.0
        return 40.0, 100.0

    def _refresh_frequency_range(self):
        low_khz, high_khz = self._selected_freq_range()
        for ax in self.axes.values():
            ax.set_ylim(low_khz, high_khz)
        self._update_plot(force=True)

    def _parse_config(self):
        fs = int(float(self.sample_rate_var.get()))
        chunk = int(float(self.chunk_var.get()))
        display_sec = float(self.display_sec_var.get())
        if fs < 10_000:
            raise ValueError("取樣率必須 >= 10,000 Hz")
        if chunk < 1024:
            raise ValueError("每次讀取點數必須 >= 1024")
        if display_sec <= 0:
            raise ValueError("累積保留秒數必須大於 0")
        device = self.device_var.get().strip()
        channels = [
            self._full_channel_name(device, self.ai0_var.get()),
            self._full_channel_name(device, self.ai1_var.get()),
        ]
        return fs, chunk, display_sec, channels

    def _configure_ring(self, fs, chunk, display_sec):
        capacity = max(int(fs * max(6.0, self.baseline_seconds + 1.0)), chunk * 50)
        with self.ring_lock:
            self.ring_capacity = max(capacity, chunk)
            self.ring_ai0 = np.zeros(self.ring_capacity, dtype=np.float32)
            self.ring_ai1 = np.zeros(self.ring_capacity, dtype=np.float32)
            self.ring_write = 0
            self.ring_count = 0
            self.ai0_buffer = np.empty(0, dtype=np.float32)
            self.ai1_buffer = np.empty(0, dtype=np.float32)

    def _reset_plot_state(self, visual_cache_seconds):
        self.visual_cache_seconds = float(max(1.0, visual_cache_seconds))
        self.total_samples_read = 0
        self.last_plot_time = 0.0
        self.latest_metrics = {}
        self.latest_dual_metrics = {}
        self.latest_plot_payload = None
        self.plot_cache = {"ai0": None, "ai1": None}
        self._clear_trend(redraw=True)
        with self.plot_lock:
            self.plot_pending_ai0 = []
            self.plot_pending_ai1 = []
            self.plot_pending_start_sample = 0
            self.plot_pending_samples = 0

    def _append_plot_pending(self, ai0, ai1, start_sample):
        ai0 = np.asarray(ai0, dtype=np.float32)
        ai1 = np.asarray(ai1, dtype=np.float32)
        n = min(ai0.size, ai1.size)
        if n <= 0:
            return
        max_pending = int(self.sample_rate * 5.0)
        with self.plot_lock:
            if not self.plot_pending_ai0:
                self.plot_pending_start_sample = int(start_sample)
            self.plot_pending_ai0.append(ai0[:n].copy())
            self.plot_pending_ai1.append(ai1[:n].copy())
            self.plot_pending_samples += n
            while self.plot_pending_samples > max_pending and self.plot_pending_ai0:
                removed0 = self.plot_pending_ai0.pop(0)
                self.plot_pending_ai1.pop(0)
                removed_n = removed0.size
                self.plot_pending_start_sample += removed_n
                self.plot_pending_samples -= removed_n

    def _consume_plot_pending(self):
        with self.plot_lock:
            if not self.plot_pending_ai0:
                return None
            ai0 = np.concatenate(self.plot_pending_ai0)
            ai1 = np.concatenate(self.plot_pending_ai1)
            start_sample = int(self.plot_pending_start_sample)
            self.plot_pending_ai0 = []
            self.plot_pending_ai1 = []
            self.plot_pending_start_sample = start_sample + min(ai0.size, ai1.size)
            self.plot_pending_samples = 0
        n = min(ai0.size, ai1.size)
        if n <= 0:
            return None
        return start_sample, ai0[:n], ai1[:n]

    def _append_spectrogram_cache(self, channel, spec):
        freqs_khz, times, db = spec
        if times.size == 0 or db.size == 0:
            return
        cache = self.plot_cache.get(channel)
        if cache is None:
            merged = {
                "freqs_khz": freqs_khz,
                "times": times,
                "raw_db": db,
            }
        else:
            cache_freqs = cache["freqs_khz"]
            if cache_freqs.shape != freqs_khz.shape or not np.allclose(cache_freqs, freqs_khz):
                db = np.vstack([
                    np.interp(cache_freqs, freqs_khz, db[:, col], left=np.nan, right=np.nan)
                    for col in range(db.shape[1])
                ]).T
                freqs_khz = cache_freqs
            merged = {
                "freqs_khz": freqs_khz,
                "times": np.concatenate([cache["times"], times]),
                "raw_db": np.concatenate([cache["raw_db"], db], axis=1),
            }
        max_time = float(merged["times"][-1])
        keep = merged["times"] >= max(0.0, max_time - self.visual_cache_seconds)
        merged["times"] = merged["times"][keep]
        merged["raw_db"] = merged["raw_db"][:, keep]
        self.plot_cache[channel] = merged

    def _append_ring(self, ai0, ai1):
        ai0 = np.asarray(ai0, dtype=np.float32)
        ai1 = np.asarray(ai1, dtype=np.float32)
        n = min(ai0.size, ai1.size)
        if n <= 0:
            return
        ai0 = ai0[-min(n, self.ring_capacity):]
        ai1 = ai1[-min(n, self.ring_capacity):]
        n = ai0.size
        with self.ring_lock:
            if n >= self.ring_capacity:
                self.ring_ai0[:] = ai0[-self.ring_capacity:]
                self.ring_ai1[:] = ai1[-self.ring_capacity:]
                self.ring_write = 0
                self.ring_count = self.ring_capacity
                return
            end = self.ring_write + n
            if end <= self.ring_capacity:
                self.ring_ai0[self.ring_write:end] = ai0
                self.ring_ai1[self.ring_write:end] = ai1
            else:
                first = self.ring_capacity - self.ring_write
                self.ring_ai0[self.ring_write:] = ai0[:first]
                self.ring_ai1[self.ring_write:] = ai1[:first]
                self.ring_ai0[:end % self.ring_capacity] = ai0[first:]
                self.ring_ai1[:end % self.ring_capacity] = ai1[first:]
            self.ring_write = end % self.ring_capacity
            self.ring_count = min(self.ring_capacity, self.ring_count + n)

    def _snapshot_recent(self, seconds=None):
        try:
            display_sec = float(self.display_sec_var.get()) if seconds is None else float(seconds)
        except Exception:
            display_sec = 0.5
        count = max(1, int(self.sample_rate * display_sec))
        with self.ring_lock:
            count = min(count, self.ring_count)
            if count <= 0:
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
            start = (self.ring_write - count) % self.ring_capacity
            if start + count <= self.ring_capacity:
                ai0 = self.ring_ai0[start:start + count].copy()
                ai1 = self.ring_ai1[start:start + count].copy()
            else:
                first = self.ring_capacity - start
                ai0 = np.concatenate([self.ring_ai0[start:], self.ring_ai0[:count - first]]).copy()
                ai1 = np.concatenate([self.ring_ai1[start:], self.ring_ai1[:count - first]]).copy()
        return ai0, ai1

    def start(self):
        if self.running:
            return
        try:
            fs, chunk, display_sec, channels = self._parse_config()
            simulation = bool(self.simulation_var.get())
            if AEsignal_DualChannelContinuousReader is None and not simulation:
                raise RuntimeError(f"nidaq_capture 連續讀取器無法使用：{NIDAQ_IMPORT_ERROR}")
            self.sample_rate = fs
            self.chunk_size = chunk
            self._configure_ring(fs, chunk, display_sec)
            self._reset_plot_state(display_sec)
            self.stop_event.clear()
            self.suppress_reader_errors = False
            self.error_popup_shown = False
            acq = parse_acq_controls(self.acq_range_var.get(), self.acq_coupling_var.get(), self.acq_terminal_var.get())
            self.reader = AEsignal_DualChannelContinuousReader(
                samplerate=fs,
                channelnames=channels,
                chunk_size=chunk,
                simulation=simulation,
                daq_buffer_seconds=5.0,
                **acq,
            ).start()
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            mode_text = "模擬" if simulation else "硬體"
            cp_txt = acq["coupling"] or "DC"
            self.status_var.set(
                f"監看中，未錄製，不會寫入原始資料（{mode_text}）：{channels[0]} / {channels[1]}，"
                f"取樣率 {fs:,} Hz，量程 ±{acq['max_val']}V，{cp_txt} 耦合")
            self.worker_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.worker_thread.start()
        except Exception as exc:
            ErrorHandler.log_error(exc, "start realtime AE spectrum")
            messagebox.showerror("即時 AE Spectrum", str(exc))
            self.status_var.set(f"啟動失敗：{exc}")
            self._stop_reader(quiet=True, status_text="已停止")

    def stop(self):
        self._stop_reader(quiet=True, status_text="已停止")

    def _stop_reader(self, quiet=True, status_text=None):
        self.suppress_reader_errors = bool(quiet)
        self.stop_event.set()
        self.running = False
        reader = self.reader
        self.reader = None
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        if self.worker_thread is not None and self.worker_thread.is_alive() and threading.current_thread() is not self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        self.worker_thread = None
        if self.recording:
            self.toggle_record()
        with self.detect_lock:
            self.detect_enabled = False
            self.detect_latest = None
            self.detect_acc0 = np.empty(0, dtype=np.float64)
            self.detect_acc1 = np.empty(0, dtype=np.float64)
            self.detect_band_sos = None  # 需依新 sample_rate 重建
            self.detect_win_n = 0
        self.detect_status_var.set("未校準")
        try:
            self.detect_status_label.configure(foreground="gray")
        except Exception:
            pass
        self.detect_rms_var.set("--"); self.detect_cusum_var.set("--")
        self.detect_burst_var.set("--"); self.detect_res_var.set("--")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if status_text is not None:
            self.status_var.set(status_text)
        self.suppress_reader_errors = False

    # ----- 洩漏偵測 -----
    def _ensure_detect_filter(self):
        if self.detect_band_sos is None or self.detect_win_n == 0:
            nyq = self.sample_rate / 2.0
            hi = min(self.detect_band_hi, nyq * 0.95)
            self.detect_band_sos = signal.butter(
                4, [self.detect_band_lo, hi], btype="bandpass", fs=self.sample_rate, output="sos")
            self.detect_win_n = int(self.detect_win_sec * self.sample_rate)

    def _compute_window_rms_series(self, x, sos, win_n):
        if x.size < win_n + 50:
            return np.array([])
        y = signal.sosfiltfilt(sos, x.astype(np.float64))
        nf = y.size // win_n
        if nf == 0:
            return np.array([])
        yt = y[: nf * win_n].reshape(nf, win_n)
        return np.sqrt(np.mean(yt ** 2, axis=1))

    def _calibrate_detectors_from_data(self, ai0, ai1):
        """用一段正常資料（raw ai0/ai1）擬合四個 detector。回傳 (ok, info_str)。"""
        if not LEAK_DETECTORS_AVAILABLE:
            return False, "leak_detectors 模組不可用"
        try:
            self._ensure_detect_filter()
        except Exception as exc:
            return False, f"建立帶通濾波器失敗：{exc}"
        sos, win_n = self.detect_band_sos, self.detect_win_n
        rms_s = self._compute_window_rms_series(ai0, sos, win_n)
        rms_r = self._compute_window_rms_series(ai1, sos, win_n)
        n = min(rms_s.size, rms_r.size)
        if n < 30:
            return False, f"視窗不夠 (n={n}，需 ≥30)"
        rms_s = rms_s[:n]; rms_r = rms_r[:n]
        mu_s = float(rms_s.mean()); sd_s = float(rms_s.std())
        med_s, mad_s = median_mad(rms_s)
        with self.detect_lock:
            self.detect_adaptive = AdaptiveThresholdDetector(n_keep=600, k=5.0, init_n=n)
            for v in rms_s:
                self.detect_adaptive.update(float(v))
            self.detect_cusum = CUSUMDetector(mu_s, sd_s, k_fac=0.5, h_fac=4.0)
            self.detect_burst = BurstRateDetector(burst_thr=med_s + 5 * mad_s,
                                                   window_n=100, trigger_rate=0.3)
            self.detect_refsub_obj = ReferenceSubtractor(rms_s, rms_r)
            self.detect_refsub_adaptive = AdaptiveThresholdDetector(n_keep=600, k=5.0, init_n=n)
            for sv, rv in zip(rms_s, rms_r):
                e = self.detect_refsub_obj.transform(float(sv), float(rv))
                self.detect_refsub_adaptive.update(e)
            self.detect_acc0 = np.empty(0, dtype=np.float64)
            self.detect_acc1 = np.empty(0, dtype=np.float64)
            self.detect_latest = None
            self.detect_enabled = True
        self.detect_status_var.set("正常")
        try:
            self.detect_status_label.configure(foreground="green")
        except Exception:
            pass
        info = (f"μ={mu_s*1e3:.3f}mV σ={sd_s*1e3:.3f}mV  "
                f"median={med_s*1e3:.3f}mV MAD={mad_s*1e3:.3f}mV  "
                f"Burst門檻={(med_s+5*mad_s)*1e3:.3f}mV  "
                f"RefSub β={self.detect_refsub_obj.beta:.3f}  (n={n} 窗)")
        return True, info

    def _calibrate_detectors(self):
        """按鈕回呼：用目前 ring buffer 重新校準（不動頻譜 baseline）。"""
        if not self.running:
            messagebox.showinfo("洩漏偵測", "請先按「開始」啟動擷取。")
            return
        if self.ai0_buffer.size < int(self.sample_rate * 3):
            messagebox.showinfo("洩漏偵測", "需至少 3 秒資料；多收一點再校準。")
            return
        ok, info = self._calibrate_detectors_from_data(self.ai0_buffer, self.ai1_buffer)
        if ok:
            messagebox.showinfo("洩漏偵測", "重新校準完成。\n" + info)
        else:
            messagebox.showwarning("洩漏偵測", info)

    def _reset_cusum(self):
        with self.detect_lock:
            if self.detect_cusum is not None:
                self.detect_cusum.S = 0.0

    def _process_detector_chunk(self, ai0_new, ai1_new):
        if not self.detect_enabled or self.detect_band_sos is None:
            return
        with self.detect_lock:
            if not self.detect_enabled:
                return
            self.detect_acc0 = np.concatenate([self.detect_acc0, ai0_new.astype(np.float64)])
            self.detect_acc1 = np.concatenate([self.detect_acc1, ai1_new.astype(np.float64)])
            win_n = self.detect_win_n
            sos = self.detect_band_sos
            while self.detect_acc0.size >= win_n:
                seg0 = self.detect_acc0[:win_n]; seg1 = self.detect_acc1[:win_n]
                try:
                    bp0 = signal.sosfiltfilt(sos, seg0)
                    bp1 = signal.sosfiltfilt(sos, seg1)
                except Exception:
                    self.detect_acc0 = self.detect_acc0[win_n:]
                    self.detect_acc1 = self.detect_acc1[win_n:]
                    continue
                rms0 = float(np.sqrt(np.mean(bp0 ** 2)))
                rms1 = float(np.sqrt(np.mean(bp1 ** 2)))
                t_a, thr_a = self.detect_adaptive.update(rms0)
                t_c, sval = self.detect_cusum.update(rms0)
                t_b, rate = self.detect_burst.update(rms0)
                e = self.detect_refsub_obj.transform(rms0, rms1)
                t_r, _ = self.detect_refsub_adaptive.update(e)
                self.detect_latest = {
                    "rms": rms0, "rms_r": rms1,
                    "cusum_S": sval, "burst_rate": rate, "residual": e,
                    "adaptive_thr": thr_a if thr_a is not None and np.isfinite(thr_a) else float("nan"),
                    "triggers": {"adaptive": bool(t_a), "cusum": bool(t_c),
                                 "burst": bool(t_b), "refsub": bool(t_r)},
                }
                self.detect_acc0 = self.detect_acc0[win_n:]
                self.detect_acc1 = self.detect_acc1[win_n:]

    def _update_detector_ui(self):
        if not self.detect_enabled or self.detect_latest is None:
            return
        d = self.detect_latest
        trigs = d["triggers"]
        on = [k for k, v in trigs.items() if v]
        if on:
            self.detect_status_var.set("洩漏 [" + "+".join(on) + "]")
            self.detect_status_label.configure(foreground="red")
        else:
            self.detect_status_var.set("正常")
            self.detect_status_label.configure(foreground="green")
        thr_txt = f"(thr {d['adaptive_thr']*1e3:.2f})" if np.isfinite(d["adaptive_thr"]) else ""
        self.detect_rms_var.set(f"{d['rms']*1e3:.3f} mV {thr_txt}")
        self.detect_cusum_var.set(f"{d['cusum_S']*1e3:.2f} mV·窗")
        self.detect_burst_var.set(f"{d['burst_rate']*100:.0f}%")
        self.detect_res_var.set(f"{d['residual']*1e3:+.3f} mV")

    def _reader_loop(self):
        while not self.stop_event.is_set():
            try:
                reader = self.reader
                if reader is None:
                    break
                chunk = reader.read_chunk()
                ai0 = np.asarray(chunk.get("ai0", []), dtype=np.float32)
                ai1 = np.asarray(chunk.get("ai1", []), dtype=np.float32)
                n = min(ai0.size, ai1.size)
                if n <= 0:
                    continue
                ai0 = ai0[:n]
                ai1 = ai1[:n]
                start_sample = self.total_samples_read
                self.total_samples_read += n
                self._append_ring(ai0, ai1)
                self._process_detector_chunk(ai0, ai1)
                self._append_plot_pending(ai0, ai1, start_sample)
                if self.recording:
                    self._accumulate_record(ai0, ai1)
            except Exception as exc:
                if self.stop_event.is_set() or self.suppress_reader_errors:
                    break
                self.parent.after(0, lambda e=exc: self._reader_failed(e))
                break

    def _reader_failed(self, exc):
        if self.error_popup_shown:
            return
        self.error_popup_shown = True
        ErrorHandler.log_error(exc, "realtime reader loop")
        message = self._format_reader_error(exc)
        self.status_var.set(message)
        messagebox.showerror("即時 AE Spectrum", message)
        self._stop_reader(quiet=True)

    def _format_reader_error(self, exc):
        text = str(exc)
        if "-200279" in text:
            return (
                "DAQ 讀取追不上硬體擷取（NI 錯誤 -200279）。\n"
                "已停止即時讀取，避免錯誤連續彈出。\n"
                "建議：維持大 buffer、減少顯示秒數，或先用較低取樣率確認硬體穩定。"
            )
        return f"DAQ 讀取中斷：{text}"

    def _schedule_poll(self):
        self._poll_ui()
        self.parent.after(100, self._schedule_poll)

    def _poll_ui(self):
        if self.recording:
            with self.record_lock:
                ready_to_flush = self.record_samples >= self.sample_rate
            if ready_to_flush:
                self._flush_record(force=False)
        if not self.running:
            return
        now = time.time()
        if now - self.last_plot_time >= 0.8:
            self._update_plot()
            self.last_plot_time = now
        self._update_detector_ui()

    def _safe_float(self, var, default):
        try:
            return float(var.get())
        except Exception:
            return default

    def _sync_denoise_config(self):
        self.ema_alpha = min(max(self._safe_float(self.ema_alpha_var, 0.3), 0.01), 1.0)
        self.z_color_limit = max(self._safe_float(self.z_limit_var, 6.0), 0.5)
        self.coh_gate_enabled = bool(self.coh_gate_var.get())
        self.coh_gate_threshold = min(max(self._safe_float(self.coh_thr_var, 0.3), 0.0), 1.0)
        self.spec_sub_enabled = bool(self.spec_sub_var.get())
        self.spec_sub_alpha = max(self._safe_float(self.spec_alpha_var, 1.5), 0.0)
        self.spec_sub_floor = min(max(self._safe_float(self.spec_floor_var, 0.1), 0.0), 1.0)
        self.wavelet_enabled = bool(self.wavelet_var.get()) and PYWT_AVAILABLE
        self.wavelet_name = self.wavelet_name_var.get().strip() or "db6"
        self.wavelet_level = int(max(self._safe_float(self.wavelet_level_var, 4), 1))
        self.wavelet_k = max(self._safe_float(self.wavelet_k_var, 1.0), 0.0)

    def _denoise_timeseries(self, x):
        """選配時域去噪：小波軟門檻。未啟用或 pywt 不可用時原樣返回。"""
        if not self.wavelet_enabled or pywt is None:
            return x
        try:
            arr = np.asarray(x, dtype=np.float64)
            if arr.size < 2 ** (self.wavelet_level + 1):
                return x
            max_level = pywt.dwt_max_level(arr.size, pywt.Wavelet(self.wavelet_name).dec_len)
            level = max(1, min(self.wavelet_level, max_level))
            coeffs = pywt.wavedec(arr, self.wavelet_name, level=level)
            detail = coeffs[-1]
            sigma = float(np.median(np.abs(detail))) / 0.6745 if detail.size else 0.0
            if sigma <= 0:
                return x
            thr = self.wavelet_k * sigma * np.sqrt(2.0 * np.log(max(arr.size, 2)))
            coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
            rec = pywt.waverec(coeffs, self.wavelet_name)
            return np.asarray(rec[:arr.size], dtype=np.float32)
        except Exception as exc:
            ErrorHandler.log_error(exc, "wavelet denoise")
            return np.asarray(x, dtype=np.float32)

    def _apply_variance_reduction(self, db):
        """沿時間軸做指數移動平均(EMA)，壓低單框 STFT 估計的隨機變異。"""
        alpha = float(self.ema_alpha)
        if not (0.0 < alpha < 1.0) or db.ndim != 2 or db.shape[1] < 2:
            return db
        if not np.all(np.isfinite(db)):
            return db
        b = [alpha]
        a = [1.0, -(1.0 - alpha)]
        zi = signal.lfilter_zi(b, a)
        zi_full = db[:, :1] * zi[0]
        smoothed, _ = signal.lfilter(b, a, db, axis=1, zi=zi_full)
        return smoothed.astype(db.dtype, copy=False)

    def _spectral_subtract(self, channel, freqs_khz, db):
        """線性功率域譜減法：扣掉基準噪聲地板，保留 β 倍下限。"""
        baseline = self.baseline.get(channel)
        if not self._has_baseline(channel):
            return db
        noise_db = np.interp(freqs_khz, baseline["freqs_khz"], baseline["median_db"], left=np.nan, right=np.nan)
        power = np.power(10.0, db / 10.0)
        noise_power = np.power(10.0, noise_db / 10.0)[:, None]
        clean = np.maximum(power - self.spec_sub_alpha * noise_power, self.spec_sub_floor * noise_power)
        return 10.0 * np.log10(clean + 1e-30)

    def _coherence_gate(self, freqs_khz, z_map):
        """以雙感測器 coherence 為遮罩，壓掉非聲源(非相干)頻段的 z 值。"""
        prof = self._last_coh_profile
        if prof is None:
            return z_map
        coh_freqs, coh = prof
        coh_i = np.interp(freqs_khz, coh_freqs, coh, left=0.0, right=0.0)
        mask = coh_i >= self.coh_gate_threshold
        if np.all(mask):
            return z_map
        gated = z_map.copy()
        gated[~mask, :] = 0.0
        return gated

    def _zscore_db(self, channel, freqs_khz, db):
        """robust z-score：(db - 基準中位數) / (1.4826·MAD)。無基準回 None。"""
        baseline = self.baseline.get(channel)
        if not self._has_baseline(channel):
            return None
        base_freqs = baseline["freqs_khz"]
        median = np.interp(freqs_khz, base_freqs, baseline["median_db"], left=np.nan, right=np.nan)
        mad = np.interp(freqs_khz, base_freqs, baseline.get("mad_db", baseline["median_db"] * 0.0), left=np.nan, right=np.nan)
        denom = 1.4826 * np.maximum(mad, self.z_mad_floor_db)
        return (db - median[:, None]) / denom[:, None]

    def _make_channel_baseline(self, spec, seconds):
        """以與即時相同的變異縮減處理基準頻譜，存每頻率中位數與 MAD。"""
        freqs_khz, _times, raw_db = spec
        smooth_db = self._apply_variance_reduction(raw_db)
        median = np.nanmedian(smooth_db, axis=1)
        mad = np.nanmedian(np.abs(smooth_db - median[:, None]), axis=1)
        return {
            "freqs_khz": np.asarray(freqs_khz).copy(),
            "median_db": median.astype(np.float32),
            "mad_db": mad.astype(np.float32),
            "seconds": float(seconds),
            "created_at": datetime.now().isoformat(),
        }

    def _spectrogram_db(self, data, time_offset_sec=0.0):
        x = np.asarray(data, dtype=np.float32)
        if x.size < 2048:
            return None
        x = x - float(np.mean(x))
        nperseg = min(2048, max(512, x.size // 8))
        noverlap = nperseg // 2
        freqs, times, sxx = signal.spectrogram(
            x,
            fs=float(self.sample_rate),
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            mode="psd",
        )
        times = times + float(time_offset_sec)
        mask = (freqs >= 35_000) & (freqs <= 300_000)
        if not np.any(mask):
            return None
        db = 10.0 * np.log10(sxx[mask] + 1e-30)
        freqs_khz = freqs[mask] / 1000.0
        max_freq_bins = 260
        max_time_bins = 260
        freq_step = max(1, int(np.ceil(db.shape[0] / max_freq_bins)))
        time_step = max(1, int(np.ceil(db.shape[1] / max_time_bins)))
        return freqs_khz[::freq_step], times[::time_step], db[::freq_step, ::time_step]

    def _band_mean(self, freqs_khz, db, low, high):
        mask = (freqs_khz >= low) & (freqs_khz <= high)
        if not np.any(mask):
            return float("nan")
        return float(np.nanmean(db[mask, :]))

    def _metrics(self, freqs_khz, times, db):
        idx = int(np.nanargmax(db))
        freq_idx, time_idx = np.unravel_index(idx, db.shape)
        return {
            "peak_freq_khz": float(freqs_khz[freq_idx]),
            "peak_time_sec": float(times[time_idx]),
            "peak_db": float(db[freq_idx, time_idx]),
            "band_35_45_db": self._band_mean(freqs_khz, db, 35, 45),
            "band_40_100_db": self._band_mean(freqs_khz, db, 40, 100),
            "band_100_300_db": self._band_mean(freqs_khz, db, 100, 300),
            "band_250_300_db": self._band_mean(freqs_khz, db, 250, 300),
            "band_40_100_max_db": float(np.nanmax(db[(freqs_khz >= 40) & (freqs_khz <= 100), :])) if np.any((freqs_khz >= 40) & (freqs_khz <= 100)) else float("nan"),
        }

    def _band_mean_1d(self, freqs_khz, values, low, high):
        freqs_khz = np.asarray(freqs_khz, dtype=float)
        values = np.asarray(values, dtype=float)
        mask = (freqs_khz >= low) & (freqs_khz <= high) & np.isfinite(values)
        if not np.any(mask):
            return float("nan")
        return float(np.nanmean(values[mask]))

    def _empty_dual_metrics(self):
        metrics = {
            "has_dual": False,
            "has_dual_delta": False,
            "coh_peak_freq_khz": float("nan"),
            "coh_peak": float("nan"),
            "gcc_peak_score": float("nan"),
            "gcc_delay_ms": float("nan"),
        }
        for suffix in ["35_45", "40_100", "100_300", "250_300"]:
            metrics[f"coh_{suffix}"] = float("nan")
            metrics[f"coh_delta_{suffix}"] = float("nan")
            metrics[f"cpsd_{suffix}_db"] = float("nan")
            metrics[f"cpsd_delta_{suffix}_db"] = float("nan")
        return metrics

    def _gcc_phat_metrics(self, ai0, ai1, low_hz=40_000.0, high_hz=100_000.0):
        n = min(ai0.size, ai1.size)
        if n < 2048:
            return {"gcc_peak_score": float("nan"), "gcc_delay_ms": float("nan")}
        max_samples = min(n, int(max(2048, self.sample_rate * 0.25)))
        x = np.asarray(ai0[-max_samples:], dtype=np.float64)
        y = np.asarray(ai1[-max_samples:], dtype=np.float64)
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        if not np.any(np.isfinite(x)) or not np.any(np.isfinite(y)):
            return {"gcc_peak_score": float("nan"), "gcc_delay_ms": float("nan")}
        nfft = 1 << int(np.ceil(np.log2(max(2 * x.size, 2))))
        x_fft = np.fft.rfft(x, n=nfft)
        y_fft = np.fft.rfft(y, n=nfft)
        cross = y_fft * np.conj(x_fft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / max(float(self.sample_rate), 1.0))
        high_hz = min(float(high_hz), float(self.sample_rate) * 0.49)
        band_mask = (freqs >= float(low_hz)) & (freqs <= high_hz)
        if np.any(band_mask):
            cross[~band_mask] = 0.0
        denom = np.abs(cross)
        cross = cross / (denom + 1e-20)
        corr = np.fft.irfft(cross, n=nfft)
        max_shift = min(int(0.005 * float(self.sample_rate)), nfft // 2 - 1)
        if max_shift <= 0:
            return {"gcc_peak_score": float("nan"), "gcc_delay_ms": float("nan")}
        corr = np.concatenate([corr[-max_shift:], corr[:max_shift + 1]])
        lags = np.arange(-max_shift, max_shift + 1, dtype=float) / max(float(self.sample_rate), 1.0)
        corr_abs = np.abs(corr)
        if corr_abs.size == 0 or not np.any(np.isfinite(corr_abs)):
            return {"gcc_peak_score": float("nan"), "gcc_delay_ms": float("nan")}
        peak_idx = int(np.nanargmax(corr_abs))
        median_level = float(np.nanmedian(corr_abs)) + 1e-20
        return {
            "gcc_peak_score": float(corr_abs[peak_idx] / median_level),
            "gcc_delay_ms": float(lags[peak_idx] * 1000.0),
        }

    def _compute_dual_channel_metrics(self, ai0, ai1, apply_baseline=True):
        metrics = self._empty_dual_metrics()
        try:
            n = min(np.asarray(ai0).size, np.asarray(ai1).size)
            if n < 2048:
                return metrics
            max_samples = min(n, int(max(2048, self.sample_rate * 1.0)))
            x = np.asarray(ai0[-max_samples:], dtype=np.float64)
            y = np.asarray(ai1[-max_samples:], dtype=np.float64)
            x = x - float(np.mean(x))
            y = y - float(np.mean(y))
            nperseg = min(8192, max(1024, x.size // 8))
            nperseg = min(nperseg, x.size)
            if nperseg < 256:
                return metrics
            noverlap = nperseg // 2
            freqs, coh = signal.coherence(
                x,
                y,
                fs=float(self.sample_rate),
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False,
            )
            csd_freqs, pxy = signal.csd(
                x,
                y,
                fs=float(self.sample_rate),
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False,
                scaling="density",
            )
            freqs_khz = freqs / 1000.0
            csd_freqs_khz = csd_freqs / 1000.0
            coh = np.clip(np.asarray(coh, dtype=float), 0.0, 1.0)
            self._last_coh_profile = (freqs_khz.copy(), coh.copy())
            cpsd_db = 10.0 * np.log10(np.abs(pxy) + 1e-30)
            peak_mask = (freqs_khz >= 35) & (freqs_khz <= 300)
            if np.any(peak_mask) and np.any(np.isfinite(coh[peak_mask])):
                peak_indices = np.flatnonzero(peak_mask)
                local_idx = int(np.nanargmax(coh[peak_mask]))
                peak_idx = int(peak_indices[local_idx])
                metrics["coh_peak_freq_khz"] = float(freqs_khz[peak_idx])
                metrics["coh_peak"] = float(coh[peak_idx])
            for suffix, low, high in [
                ("35_45", 35, 45),
                ("40_100", 40, 100),
                ("100_300", 100, 300),
                ("250_300", 250, 300),
            ]:
                metrics[f"coh_{suffix}"] = self._band_mean_1d(freqs_khz, coh, low, high)
                metrics[f"cpsd_{suffix}_db"] = self._band_mean_1d(csd_freqs_khz, cpsd_db, low, high)
            metrics.update(self._gcc_phat_metrics(x, y))
            metrics["has_dual"] = True
            if apply_baseline and self.dual_baseline:
                for suffix in ["35_45", "40_100", "100_300", "250_300"]:
                    metrics[f"coh_delta_{suffix}"] = metrics[f"coh_{suffix}"] - self.dual_baseline.get(f"coh_{suffix}", float("nan"))
                    metrics[f"cpsd_delta_{suffix}_db"] = metrics[f"cpsd_{suffix}_db"] - self.dual_baseline.get(f"cpsd_{suffix}_db", float("nan"))
                metrics["has_dual_delta"] = any(
                    np.isfinite(metrics.get(f"coh_delta_{suffix}", float("nan")))
                    for suffix in ["35_45", "40_100", "100_300", "250_300"]
                )
        except Exception as exc:
            ErrorHandler.log_error(exc, "compute dual-channel AE metrics")
        return metrics

    def _has_baseline(self, channel):
        baseline = self.baseline.get(channel)
        return bool(baseline and "freqs_khz" in baseline and "median_db" in baseline)

    def _update_channel_plot(self, channel, spec):
        freqs_khz, times, raw_db = spec
        # 基底：時間 EMA 變異縮減（永遠開）
        smooth_db = self._apply_variance_reduction(raw_db)
        # 選配：譜減法（線性功率域扣噪聲地板）
        proc_db = smooth_db
        if self.spec_sub_enabled and self._has_baseline(channel):
            proc_db = self._spectral_subtract(channel, freqs_khz, smooth_db)
        # robust z-score（升級後的 Delta 內容）
        z_map = self._zscore_db(channel, freqs_khz, proc_db)
        if z_map is not None and self.coh_gate_enabled:
            z_map = self._coherence_gate(freqs_khz, z_map)
        has_delta = z_map is not None
        delta_metrics = self._metrics(freqs_khz, times, z_map) if has_delta else None
        use_delta = self._is_delta_mode() and has_delta
        display_source = z_map if use_delta else smooth_db
        low_khz, high_khz = self._selected_freq_range()
        display_mask = (freqs_khz >= low_khz) & (freqs_khz <= high_khz)
        if not np.any(display_mask):
            display_mask = np.ones_like(freqs_khz, dtype=bool)
        display_freqs = freqs_khz[display_mask]
        plot_db = display_source[display_mask, :]
        display_db = plot_db
        if self.smooth_display_var.get() and display_db.shape[0] >= 3 and display_db.shape[1] >= 3:
            display_db = ndimage.median_filter(display_db, size=(3, 3), mode="nearest")
        im = self.images[channel]
        ax = self.axes[channel]
        marker = self.peak_markers[channel]
        if use_delta:
            vmin, vmax = -self.z_color_limit, self.z_color_limit
            cmap = "coolwarm"
            color_label = "相對正常 z-score (σ)"
            title_suffix = "相對正常 z-score (σ)"
        else:
            vmin, vmax = np.nanpercentile(display_db, [2, 99])
            cmap = "magma"
            color_label = "dB（Raw 訊號品質檢查）"
            title_suffix = "Raw dB（變異縮減後）"
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = (-self.z_color_limit, self.z_color_limit) if use_delta else (-120.0, 0.0)
        extent = [float(times[0]), float(times[-1]) if times.size else 1.0, float(display_freqs[0]), float(display_freqs[-1])]
        im.set_data(display_db)
        im.set_extent(extent)
        im.set_clim(vmin, vmax)
        im.set_cmap(cmap)
        self.colorbars[channel].set_label(color_label)
        raw_metrics = self._metrics(freqs_khz, times, smooth_db)
        display_metrics = self._metrics(display_freqs, times, plot_db)
        if use_delta:
            marker.set_offsets(np.asarray([[display_metrics["peak_time_sec"], display_metrics["peak_freq_khz"]]], dtype=float))
        else:
            marker.set_offsets(np.empty((0, 2)))
        ax.set_title(f"{channel} 感測器 Spectrum - {title_suffix}")
        ax.set_xlim(extent[0], max(extent[1], extent[0] + 0.01))
        ax.set_ylim(low_khz, high_khz)
        metrics = dict(delta_metrics) if delta_metrics is not None else dict(raw_metrics)
        metrics.update({
            "raw_peak_freq_khz": raw_metrics["peak_freq_khz"],
            "raw_peak_db": raw_metrics["peak_db"],
            "has_delta": bool(has_delta),
        })
        if delta_metrics is not None:
            metrics.update({
                "delta_peak_freq_khz": delta_metrics["peak_freq_khz"],
                "delta_peak_time_sec": delta_metrics["peak_time_sec"],
                "delta_peak_db": delta_metrics["peak_db"],
                "delta_band_35_45_db": delta_metrics["band_35_45_db"],
                "delta_band_40_100_db": delta_metrics["band_40_100_db"],
                "delta_band_40_100_max_db": delta_metrics["band_40_100_max_db"],
                "delta_band_100_300_db": delta_metrics["band_100_300_db"],
                "delta_band_250_300_db": delta_metrics["band_250_300_db"],
            })
        else:
            metrics.update({
                "delta_peak_freq_khz": float("nan"),
                "delta_peak_time_sec": float("nan"),
                "delta_peak_db": float("nan"),
                "delta_band_35_45_db": float("nan"),
                "delta_band_40_100_db": float("nan"),
                "delta_band_40_100_max_db": float("nan"),
                "delta_band_100_300_db": float("nan"),
                "delta_band_250_300_db": float("nan"),
            })
        return metrics

    def _update_plot(self, force=False):
        self._sync_denoise_config()
        pending = self._consume_plot_pending()
        if pending is not None:
            start_sample, ai0, ai1 = pending
            self.ai0_buffer = ai0
            self.ai1_buffer = ai1
            time_offset_sec = start_sample / max(float(self.sample_rate), 1.0)
            ai0_proc = self._denoise_timeseries(ai0)
            ai1_proc = self._denoise_timeseries(ai1)
            ai0_spec = self._spectrogram_db(ai0_proc, time_offset_sec=time_offset_sec)
            ai1_spec = self._spectrogram_db(ai1_proc, time_offset_sec=time_offset_sec)
            if ai0_spec is not None:
                self._append_spectrogram_cache("ai0", ai0_spec)
            if ai1_spec is not None:
                self._append_spectrogram_cache("ai1", ai1_spec)
        ai0_spec = self.plot_cache.get("ai0")
        ai1_spec = self.plot_cache.get("ai1")
        if ai0_spec is None or ai1_spec is None:
            return
        # 先算雙通道指標（順帶快取 coherence profile，供相干遮罩使用）
        self.latest_dual_metrics = self._compute_dual_channel_metrics(self.ai0_buffer, self.ai1_buffer)
        ai0_tuple = (ai0_spec["freqs_khz"], ai0_spec["times"], ai0_spec["raw_db"])
        ai1_tuple = (ai1_spec["freqs_khz"], ai1_spec["times"], ai1_spec["raw_db"])
        m0 = self._update_channel_plot("ai0", ai0_tuple)
        m1 = self._update_channel_plot("ai1", ai1_tuple)
        self.latest_metrics = {"ai0": m0, "ai1": m1}
        self.latest_plot_payload = {"ai0": ai0_tuple, "ai1": ai1_tuple}
        self._update_metric_labels()
        elapsed = max(float(ai0_spec["times"][-1]), float(ai1_spec["times"][-1]))
        self._append_trend_point(elapsed)
        self._update_trend_plot(elapsed)
        self.canvas.draw_idle()
        if self.recording:
            self.status_var.set(f"錄製中，正在保存原始 ai0/ai1；累積顯示 {elapsed:.1f} 秒")
        elif self.running:
            self.status_var.set(f"監看中，未錄製，不會寫入原始資料；累積顯示 {elapsed:.1f} 秒")

    def _format_db(self, value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:+.2f} dB"

    def _format_sigma(self, value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:+.2f} σ"

    def _format_plain_db(self, value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.1f} dB"

    def _format_unit_delta(self, value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:+.3f}"

    def _format_score(self, value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:.1f}x"

    def _format_ms(self, value):
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:+.3f}"

    def _set_indicator_color(self, channel, name, value):
        widget = self.metric_widgets.get(channel, {}).get(name)
        if widget is None:
            return
        if value is None or not np.isfinite(value):
            widget.configure(foreground="gray")
        elif value >= self.delta_alert_db:
            widget.configure(foreground="#c0392b")
        elif value >= self.delta_warn_db:
            widget.configure(foreground="#d35400")
        elif value <= -self.delta_alert_db:
            widget.configure(foreground="#21618c")
        else:
            widget.configure(foreground="gray")

    def _set_dual_indicator_color(self, name, value, warn=0.15, alert=0.30):
        widget = self.dual_metric_widgets.get(name)
        if widget is None:
            return
        if value is None or not np.isfinite(value):
            widget.configure(foreground="gray")
        elif value >= alert:
            widget.configure(foreground="#c0392b")
        elif value >= warn:
            widget.configure(foreground="#d35400")
        elif value <= -alert:
            widget.configure(foreground="#21618c")
        else:
            widget.configure(foreground="gray")

    def _status_from_delta(self, metrics):
        if not metrics.get("has_delta"):
            return "請先設定基準", "gray"
        values = [
            metrics.get("delta_band_40_100_db", float("nan")),
            metrics.get("delta_band_40_100_max_db", float("nan")),
            metrics.get("delta_band_100_300_db", float("nan")),
            metrics.get("delta_band_250_300_db", float("nan")),
            metrics.get("delta_band_35_45_db", float("nan")),
            metrics.get("delta_peak_db", float("nan")),
        ]
        finite = [v for v in values if np.isfinite(v)]
        score = max(finite) if finite else float("nan")
        if not np.isfinite(score):
            return "--", "gray"
        if score >= self.delta_alert_db:
            return "明顯高於正常", "#c0392b"
        if score >= self.delta_warn_db:
            return "略高於正常", "#d35400"
        if score <= -self.delta_alert_db:
            return "低於正常", "#21618c"
        return "接近正常", "gray"

    def _dual_status_from_metrics(self, metrics):
        if not metrics.get("has_dual"):
            return "--", "gray"
        if not metrics.get("has_dual_delta"):
            return "請先設定基準", "gray"
        coh40 = metrics.get("coh_delta_40_100", float("nan"))
        cpsd40 = metrics.get("cpsd_delta_40_100_db", float("nan"))
        gcc_score = metrics.get("gcc_peak_score", float("nan"))
        coh_values = [
            metrics.get("coh_delta_35_45", float("nan")),
            metrics.get("coh_delta_40_100", float("nan")),
            metrics.get("coh_delta_100_300", float("nan")),
            metrics.get("coh_delta_250_300", float("nan")),
        ]
        finite_coh = [value for value in coh_values if np.isfinite(value)]
        if not finite_coh:
            return "--", "gray"
        max_coh = max(finite_coh)
        strong_cpsd = np.isfinite(cpsd40) and cpsd40 >= 0.8
        weak_cpsd = np.isfinite(cpsd40) and cpsd40 >= 0.4
        strong_gcc = np.isfinite(gcc_score) and gcc_score >= 8.0
        weak_gcc = np.isfinite(gcc_score) and gcc_score >= 5.0
        if (max_coh >= 0.30 and strong_cpsd) or (coh40 >= 0.20 and strong_gcc):
            return "雙通道明顯相干", "#c0392b"
        if max_coh >= 0.15 or weak_cpsd or weak_gcc:
            return "雙通道可疑", "#d35400"
        if max_coh <= -0.30:
            return "低於基準相干", "#21618c"
        return "接近基準", "gray"

    def _update_metric_labels(self):
        for channel in ["ai0", "ai1"]:
            metrics = self.latest_metrics.get(channel, {})
            if not metrics:
                continue
            peak_freq = metrics.get("delta_peak_freq_khz")
            peak_db = metrics.get("delta_peak_db")
            band_35_45 = metrics.get("delta_band_35_45_db")
            band_40_100 = metrics.get("delta_band_40_100_db")
            band_40_100_max = metrics.get("delta_band_40_100_max_db")
            band_100_300 = metrics.get("delta_band_100_300_db")
            band_250_300 = metrics.get("delta_band_250_300_db")
            self.metric_vars[channel]["peak"].set(f"{peak_freq:.1f} kHz" if np.isfinite(peak_freq) else "--")
            self.metric_vars[channel]["peak_db"].set(self._format_sigma(peak_db))
            self.metric_vars[channel]["band_35_45"].set(self._format_sigma(band_35_45))
            self.metric_vars[channel]["band_40_100"].set(self._format_sigma(band_40_100))
            self.metric_vars[channel]["band_40_100_max"].set(self._format_sigma(band_40_100_max))
            self.metric_vars[channel]["band_100_300"].set(self._format_sigma(band_100_300))
            self.metric_vars[channel]["band_250_300"].set(self._format_sigma(band_250_300))
            status_text, status_color = self._status_from_delta(metrics)
            self.metric_vars[channel]["status"].set(status_text)
            self.metric_widgets[channel]["status"].configure(foreground=status_color)
            self._set_indicator_color(channel, "peak_db", peak_db)
            self._set_indicator_color(channel, "band_35_45", band_35_45)
            self._set_indicator_color(channel, "band_40_100", band_40_100)
            self._set_indicator_color(channel, "band_40_100_max", band_40_100_max)
            self._set_indicator_color(channel, "band_100_300", band_100_300)
            self._set_indicator_color(channel, "band_250_300", band_250_300)
        n = min(self.ai0_buffer.size, self.ai1_buffer.size)
        diff_rms = float("nan")
        if n:
            diff = self.ai1_buffer[-n:] - self.ai0_buffer[-n:]
            diff_rms = float(np.sqrt(np.mean(diff * diff)))
        ai0_metrics = self.latest_metrics.get("ai0", {})
        ai1_metrics = self.latest_metrics.get("ai1", {})
        ai0_peak = ai0_metrics.get("delta_peak_freq_khz")
        ai1_peak = ai1_metrics.get("delta_peak_freq_khz")
        peak_gap = f"{abs(ai1_peak - ai0_peak):.1f} kHz" if np.isfinite(ai0_peak) and np.isfinite(ai1_peak) else "--"
        band100_gap = self._format_sigma(ai1_metrics.get("delta_band_100_300_db", float("nan")) - ai0_metrics.get("delta_band_100_300_db", float("nan")))
        band250_gap = self._format_sigma(ai1_metrics.get("delta_band_250_300_db", float("nan")) - ai0_metrics.get("delta_band_250_300_db", float("nan")))
        band40_100_gap = self._format_sigma(ai1_metrics.get("delta_band_40_100_db", float("nan")) - ai0_metrics.get("delta_band_40_100_db", float("nan")))
        self.diff_rms_var.set(
            f"ai1-ai0 RMS: {diff_rms:.4g} | z 峰值頻率差: {peak_gap} | "
            f"40-100k 通道差(ai1-ai0): {band40_100_gap} | 100-300k: {band100_gap} | 250-300k: {band250_gap}"
        )
        self._update_dual_metric_labels()

    def _update_dual_metric_labels(self):
        metrics = self.latest_dual_metrics or {}
        coh35 = metrics.get("coh_delta_35_45", float("nan"))
        coh40 = metrics.get("coh_delta_40_100", float("nan"))
        coh100 = metrics.get("coh_delta_100_300", float("nan"))
        coh250 = metrics.get("coh_delta_250_300", float("nan"))
        cpsd40 = metrics.get("cpsd_delta_40_100_db", float("nan"))
        gcc_score = metrics.get("gcc_peak_score", float("nan"))
        delay_ms = metrics.get("gcc_delay_ms", float("nan"))
        self.dual_metric_vars["coh_35_45"].set(self._format_unit_delta(coh35))
        self.dual_metric_vars["coh_40_100"].set(self._format_unit_delta(coh40))
        self.dual_metric_vars["coh_100_300"].set(self._format_unit_delta(coh100))
        self.dual_metric_vars["coh_250_300"].set(self._format_unit_delta(coh250))
        self.dual_metric_vars["cpsd_40_100"].set(self._format_db(cpsd40))
        self.dual_metric_vars["gcc_score"].set(self._format_score(gcc_score))
        self.dual_metric_vars["delay_ms"].set(self._format_ms(delay_ms))
        status_text, status_color = self._dual_status_from_metrics(metrics)
        self.dual_metric_vars["status"].set(status_text)
        self.dual_metric_widgets["status"].configure(foreground=status_color)
        self._set_dual_indicator_color("coh_35_45", coh35)
        self._set_dual_indicator_color("coh_40_100", coh40)
        self._set_dual_indicator_color("coh_100_300", coh100)
        self._set_dual_indicator_color("coh_250_300", coh250)
        self._set_dual_indicator_color("cpsd_40_100", cpsd40, warn=0.4, alert=0.8)
        if np.isfinite(gcc_score):
            widget = self.dual_metric_widgets.get("gcc_score")
            if widget is not None:
                if gcc_score >= 8.0:
                    widget.configure(foreground="#c0392b")
                elif gcc_score >= 5.0:
                    widget.configure(foreground="#d35400")
                else:
                    widget.configure(foreground="gray")

    def set_baseline(self):
        if self.recording:
            messagebox.showwarning("設定正常基準", "錄製中不能設定正常基準，請先停止錄製。")
            return
        was_running = self.running
        ai0, ai1 = self._snapshot_recent(self.baseline_seconds)
        min_samples = int(self.sample_rate * self.baseline_min_seconds)
        if ai0.size < min_samples or ai1.size < min_samples:
            messagebox.showwarning(
                "設定正常基準",
                f"需要至少 {self.baseline_min_seconds:.0f} 秒正常資料才能設定基準；目前約 {min(ai0.size, ai1.size) / max(self.sample_rate, 1):.1f} 秒。",
            )
            return
        if was_running:
            self._stop_reader(quiet=True, status_text="設定正常基準中，DAQ 已短暫暫停...")
        try:
            self._sync_denoise_config()
            ai0_spec = self._spectrogram_db(self._denoise_timeseries(ai0))
            ai1_spec = self._spectrogram_db(self._denoise_timeseries(ai1))
            if ai0_spec is None or ai1_spec is None:
                raise RuntimeError("目前資料不足，無法建立正常基準。")
            self.baseline["ai0"] = self._make_channel_baseline(ai0_spec, float(ai0.size / max(self.sample_rate, 1)))
            self.baseline["ai1"] = self._make_channel_baseline(ai1_spec, float(ai1.size / max(self.sample_rate, 1)))
            self.dual_baseline = self._compute_dual_channel_metrics(ai0, ai1, apply_baseline=False)
            self.mode_var.set(self.delta_mode_label)
            self._clear_trend(redraw=True)
            baseline_sec = min(ai0.size, ai1.size) / max(self.sample_rate, 1)
            self.status_var.set(f"已用最近 {baseline_sec:.1f} 秒建立 ai0 / ai1 正常基準與雙通道 coherence/CPSD 基準。")
            # 同步校準洩漏偵測器（同一段正常資料）
            det_ok, det_info = self._calibrate_detectors_from_data(ai0, ai1)
            if det_ok:
                hint = ("z-score + coherence Δ 判讀：單通道 z 上升且雙通道相干也上升時，微洩漏可信度較高。  "
                        f"已同步校準洩漏偵測器：{det_info}")
            else:
                hint = ("z-score + coherence Δ 判讀：單通道 z 上升且雙通道相干也上升時，微洩漏可信度較高。  "
                        f"洩漏偵測器校準略過：{det_info}")
            self.baseline_hint_var.set(hint)
        except Exception as exc:
            ErrorHandler.log_error(exc, "set realtime baseline")
            messagebox.showerror("設定正常基準", str(exc))
            self.status_var.set(f"設定正常基準失敗：{exc}")
        finally:
            if was_running:
                self.start()
                self.status_var.set("已建立正常基準並重新開始擷取；目前以 z-score + coherence Δ 作為微洩漏主判讀。")

    def _session_dir(self):
        if self.record_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_dir = PROGRAM_DIR / "realtime_spectrum_events" / f"{stamp}_session"
            self.record_dir.mkdir(parents=True, exist_ok=True)
        return self.record_dir

    def toggle_record(self):
        if not self.recording:
            self._session_dir()
            with self.record_lock:
                self.record_buffers = []
                self.record_samples = 0
                self.record_index = 0
            self.recording = True
            self.record_btn.config(text="停止錄製")
            self.status_var.set(f"錄製中，正在保存原始 ai0/ai1 到 {self.record_dir}")
        else:
            self._flush_record(force=True)
            self.recording = False
            self.record_btn.config(text="錄製")
            self.status_var.set(f"錄製停止：{self.record_dir}；目前只監看，不會寫入原始資料")

    def _accumulate_record(self, ai0, ai1):
        with self.record_lock:
            self.record_buffers.append((np.asarray(ai0, dtype=np.float32).copy(), np.asarray(ai1, dtype=np.float32).copy()))
            self.record_samples += min(ai0.size, ai1.size)

    def _flush_record(self, force=False):
        with self.record_lock:
            if not self.record_buffers or (not force and self.record_samples < self.sample_rate):
                return
            buffers = self.record_buffers
            self.record_buffers = []
            self.record_samples = 0
            index = self.record_index
            self.record_index += 1
        ai0 = np.concatenate([pair[0] for pair in buffers])
        ai1 = np.concatenate([pair[1] for pair in buffers])
        n = min(ai0.size, ai1.size)
        path = self._session_dir() / f"record_{index:04d}.npz"
        np.savez_compressed(
            path,
            ai0=ai0[:n].astype(np.float32),
            ai1=ai1[:n].astype(np.float32),
            sample_rate=np.asarray([self.sample_rate], dtype=np.float64),
            created_at=np.asarray([datetime.now().isoformat()]),
        )

    def save_screenshot(self):
        out_dir = self._session_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = out_dir / f"spectrum_{stamp}.png"
        csv_path = out_dir / f"spectrum_metrics_{stamp}.csv"
        self.fig.savefig(png_path, dpi=160)
        self._write_metrics_csv(csv_path)
        self.status_var.set(f"已儲存截圖：{png_path}")

    def _write_metrics_csv(self, path):
        rows = []
        for channel, metrics in self.latest_metrics.items():
            row = {"timestamp": datetime.now().isoformat(), "channel": channel, "mode": self.mode_var.get()}
            row.update(metrics)
            rows.append(row)
        if self.latest_dual_metrics:
            row = {"timestamp": datetime.now().isoformat(), "channel": "dual_ai0_ai1", "mode": self.mode_var.get()}
            row.update(self.latest_dual_metrics)
            rows.append(row)
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _maybe_save_event(self):
        # 自動事件存檔已關閉；只有按「錄製」或「截圖」才會寫入資料。
        return


RealtimeAESpectrumPage = StableRealtimeAESpectrumPage


class RealtimeFeatureLightsPage:
    """即時特徵值與燈號監看頁，讀取即時 Spectrum 頁已計算好的特徵。"""

    LIGHT_COLORS = {
        "red": "#d32f2f",
        "green": "#2e7d32",
        "gray": "#9e9e9e",
    }

    def __init__(self, parent, spectrum_page):
        self.parent = parent
        self.spectrum_page = spectrum_page
        self.history_seconds = 60.0
        self.feature_history = {}
        self.rows = {}
        self.feature_specs = self._make_feature_specs()

        self.short_window_var = tk.StringVar(value="5")
        self.long_window_var = tk.StringVar(value="30")
        self.tolerance_scale_var = tk.StringVar(value="1.0")
        self.simulation_enabled = False
        self.simulation_start_time = time.time()
        self.simulation_button_text = tk.StringVar(value="模擬燈號")
        self.status_var = tk.StringVar(value="等待即時 AE Spectrum 產生特徵值。")
        self.calibration_config = None
        self.calibration_name = ""

        self._build_ui(parent)
        self._schedule_poll()

    def _make_feature_specs(self):
        specs = []
        single_features = [
            ("delta_peak_db", "最大能量 z", "sigma", 3.0, "signed"),
            ("delta_band_35_45_db", "35-45k 頻帶平均 z", "sigma", 3.0, "signed"),
            ("delta_band_40_100_db", "40-100k 頻帶平均 z", "sigma", 3.0, "signed"),
            ("delta_band_40_100_max_db", "40-100k 頻帶最大 z", "sigma", 5.0, "signed"),
            ("delta_band_100_300_db", "100-300k 頻帶平均 z", "sigma", 3.0, "signed"),
            ("delta_band_250_300_db", "250-300k 頻帶平均 z", "sigma", 3.0, "signed"),
        ]
        for channel in ["ai0", "ai1"]:
            for metric, label, unit, tolerance, mode in single_features:
                specs.append({
                    "key": f"{channel}.{metric}",
                    "section": f"{channel} 單通道特徵",
                    "tab": "時頻/STFT",
                    "category": "時頻",
                    "channel_label": channel,
                    "source": channel,
                    "metric": metric,
                    "label": label,
                    "unit": unit,
                    "default_tolerance": tolerance,
                    "mode": mode,
                    "needs_baseline": True,
                })
        dual_features = [
            ("coh_delta_35_45", "35-45k coherence", "unit", 0.15, "signed", True, "雙通道頻域", "頻域"),
            ("coh_delta_40_100", "40-100k coherence", "unit", 0.15, "signed", True, "雙通道頻域", "頻域"),
            ("coh_delta_100_300", "100-300k coherence", "unit", 0.15, "signed", True, "雙通道頻域", "頻域"),
            ("coh_delta_250_300", "250-300k coherence", "unit", 0.15, "signed", True, "雙通道頻域", "頻域"),
            ("cpsd_delta_40_100_db", "40-100k CPSD", "db", 0.4, "signed", True, "雙通道頻域", "頻域"),
            ("gcc_peak_score", "GCC-PHAT 峰值比", "score", 5.0, "high_only", True, "時域/時延", "時域"),
            ("gcc_delay_ms", "GCC-PHAT 延遲", "ms", 0.05, "signed", True, "時域/時延", "時域"),
        ]
        for metric, label, unit, tolerance, mode, needs_baseline, tab, category in dual_features:
            specs.append({
                "key": f"dual.{metric}",
                "section": "雙通道特徵",
                "tab": tab,
                "category": category,
                "channel_label": "ai0-ai1",
                "source": "dual",
                "metric": metric,
                "label": label,
                "unit": unit,
                "default_tolerance": tolerance,
                "mode": mode,
                "needs_baseline": needs_baseline,
            })
        return specs

    def _build_ui(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="燈號設定", padding=6)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=5)
        ttk.Label(controls, text="短時(s)").grid(row=0, column=0, padx=(3, 1), sticky="w")
        ttk.Entry(controls, textvariable=self.short_window_var, width=6).grid(row=0, column=1, padx=(0, 6), sticky="w")
        ttk.Label(controls, text="長時(s)").grid(row=0, column=2, padx=(3, 1), sticky="w")
        ttk.Entry(controls, textvariable=self.long_window_var, width=6).grid(row=0, column=3, padx=(0, 6), sticky="w")
        ttk.Label(controls, text="容忍倍率").grid(row=0, column=4, padx=(3, 1), sticky="w")
        ttk.Entry(controls, textvariable=self.tolerance_scale_var, width=6).grid(row=0, column=5, padx=(0, 6), sticky="w")
        ttk.Button(controls, text="套用預設", command=self._reset_tolerances).grid(row=0, column=6, padx=2)
        ttk.Button(controls, text="放寬", command=lambda: self._scale_tolerances(1.25)).grid(row=0, column=7, padx=2)
        ttk.Button(controls, text="收緊", command=lambda: self._scale_tolerances(0.8)).grid(row=0, column=8, padx=2)
        ttk.Button(controls, text="清除歷史", command=self._clear_history).grid(row=0, column=9, padx=2)
        ttk.Button(controls, textvariable=self.simulation_button_text, command=self._toggle_simulation).grid(row=0, column=10, padx=2)
        ttk.Label(controls, textvariable=self.status_var, foreground="blue").grid(row=1, column=0, columnspan=11, sticky="w", padx=3, pady=(4, 0))

        content = ttk.Notebook(parent)
        content.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        for tab_name in ["時頻/STFT", "雙通道頻域", "時域/時延"]:
            frame = ttk.Frame(content, padding=5)
            content.add(frame, text=tab_name)
            frame.columnconfigure(0, weight=1)
            self._build_feature_table(frame, tab_name)

    def _build_feature_table(self, frame, tab_name):
        headers = ["燈", "類型", "通道", "特徵名稱", "目前值/差異", "短均", "長均", "趨勢", "容忍", "狀態"]
        for col, text in enumerate(headers):
            ttk.Label(frame, text=text, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=3, pady=(0, 3), sticky="w")
        row = 1
        for spec in [item for item in self.feature_specs if tab_name == "全部特徵" or item["tab"] == tab_name]:
            light = tk.Canvas(frame, width=18, height=18, highlightthickness=0)
            oval = light.create_oval(3, 3, 15, 15, fill=self.LIGHT_COLORS["gray"], outline="#666666")
            light.grid(row=row, column=0, padx=3, pady=2)
            ttk.Label(frame, text=spec["category"], width=8).grid(row=row, column=1, padx=3, pady=2, sticky="w")
            ttk.Label(frame, text=spec["channel_label"], width=8).grid(row=row, column=2, padx=3, pady=2, sticky="w")
            ttk.Label(frame, text=spec["label"], width=20).grid(row=row, column=3, padx=3, pady=2, sticky="w")
            current_var = tk.StringVar(value="--")
            short_var = tk.StringVar(value="--")
            long_var = tk.StringVar(value="--")
            slope_var = tk.StringVar(value="--")
            tolerance_var = tk.StringVar(value=self._format_tolerance(spec["default_tolerance"]))
            status_var = tk.StringVar(value="尚無資料")
            ttk.Label(frame, textvariable=current_var, width=11).grid(row=row, column=4, padx=3, pady=2, sticky="w")
            ttk.Label(frame, textvariable=short_var, width=11).grid(row=row, column=5, padx=3, pady=2, sticky="w")
            ttk.Label(frame, textvariable=long_var, width=11).grid(row=row, column=6, padx=3, pady=2, sticky="w")
            ttk.Label(frame, textvariable=slope_var, width=11).grid(row=row, column=7, padx=3, pady=2, sticky="w")
            ttk.Entry(frame, textvariable=tolerance_var, width=7).grid(row=row, column=8, padx=3, pady=2, sticky="w")
            status_label = ttk.Label(frame, textvariable=status_var, width=12)
            status_label.grid(row=row, column=9, padx=3, pady=2, sticky="w")
            row_state = {
                "spec": spec,
                "light": light,
                "oval": oval,
                "current": current_var,
                "short": short_var,
                "long": long_var,
                "slope": slope_var,
                "tolerance": tolerance_var,
                "status": status_var,
                "status_label": status_label,
            }
            self.rows.setdefault(spec["key"], []).append(row_state)
            row += 1

    def _format_tolerance(self, value):
        return f"{float(value):.3g}"

    def _parse_float(self, value, default):
        try:
            parsed = float(str(value).strip())
            return parsed if np.isfinite(parsed) else default
        except Exception:
            return default

    def _reset_tolerances(self):
        for row_list in self.rows.values():
            for row in row_list:
                row["tolerance"].set(self._format_tolerance(row["spec"]["default_tolerance"]))
        self._refresh_all_rows()

    def _scale_tolerances(self, factor):
        for row_list in self.rows.values():
            for row in row_list:
                current = self._parse_float(row["tolerance"].get(), row["spec"]["default_tolerance"])
                row["tolerance"].set(self._format_tolerance(max(current * factor, 1e-9)))
        self._refresh_all_rows()

    def _clear_history(self):
        self.feature_history = {}
        self._refresh_all_rows()
        self.status_var.set("已清除特徵歷史，會從下一次即時更新重新累積。")

    def _toggle_simulation(self):
        self.simulation_enabled = not self.simulation_enabled
        self.simulation_start_time = time.time()
        self.simulation_button_text.set("停止模擬" if self.simulation_enabled else "模擬燈號")
        self._clear_history()
        if self.simulation_enabled:
            self.status_var.set("燈號模擬中：不讀 DAQ，只用內建特徵變化測試紅綠灰燈。")
        else:
            self.status_var.set("已停止燈號模擬，改讀即時 AE Spectrum 特徵值。")

    def _baseline_ready(self):
        if self.simulation_enabled:
            return True
        baseline = getattr(self.spectrum_page, "baseline", {})
        dual_baseline = getattr(self.spectrum_page, "dual_baseline", {})
        return bool(baseline.get("ai0") and baseline.get("ai1") and dual_baseline)

    def _feature_value(self, spec):
        if self.simulation_enabled:
            return self._simulation_value(spec)
        if spec["source"] == "dual":
            metrics = getattr(self.spectrum_page, "latest_dual_metrics", {}) or {}
        else:
            metrics = (getattr(self.spectrum_page, "latest_metrics", {}) or {}).get(spec["source"], {})
        value = metrics.get(spec["metric"], float("nan"))
        try:
            return float(value)
        except Exception:
            return float("nan")

    def _simulation_value(self, spec):
        elapsed = max(0.0, time.time() - self.simulation_start_time)
        ramp = min(max((elapsed - 3.0) / 8.0, 0.0), 1.0)
        wiggle = 0.04 * np.sin(elapsed * 2.1 + len(spec["key"]))
        channel_gain = 1.0 if spec["source"] != "ai1" else 1.25
        metric = spec["metric"]
        if metric == "delta_peak_db":
            return channel_gain * 6.0 * (0.12 + 1.05 * ramp + wiggle)
        if metric == "delta_band_35_45_db":
            return channel_gain * 6.0 * (0.03 + 0.35 * ramp + wiggle)
        if metric == "delta_band_40_100_db":
            return channel_gain * 6.0 * (0.05 + 0.85 * ramp + wiggle)
        if metric == "delta_band_40_100_max_db":
            return channel_gain * 6.0 * (0.10 + 1.20 * ramp + wiggle)
        if metric == "delta_band_100_300_db":
            return channel_gain * 6.0 * (0.02 + 0.25 * ramp + wiggle * 0.7)
        if metric == "delta_band_250_300_db":
            return channel_gain * 6.0 * (-0.03 + 0.18 * ramp + wiggle * 0.7)
        if metric == "coh_delta_35_45":
            return 0.02 + 0.10 * ramp + wiggle * 0.35
        if metric == "coh_delta_40_100":
            return 0.03 + 0.30 * ramp + wiggle * 0.35
        if metric == "coh_delta_100_300":
            return 0.01 + 0.12 * ramp + wiggle * 0.25
        if metric == "coh_delta_250_300":
            return -0.01 + 0.08 * ramp + wiggle * 0.25
        if metric == "cpsd_delta_40_100_db":
            return 0.05 + 0.95 * ramp + wiggle
        if metric == "gcc_peak_score":
            return 2.5 + 6.5 * ramp + 0.4 * np.sin(elapsed * 1.3)
        if metric == "gcc_delay_ms":
            return 0.010 + 0.025 * np.sin(elapsed * 1.1)
        return float("nan")

    def _append_history(self, key, now, value):
        if not np.isfinite(value):
            return
        values = self.feature_history.setdefault(key, [])
        values.append((float(now), float(value)))
        cutoff = float(now) - self.history_seconds
        while values and values[0][0] < cutoff:
            values.pop(0)

    def _window_stats(self, key, now, seconds):
        values = self.feature_history.get(key, [])
        cutoff = float(now) - float(seconds)
        window = [(t, v) for t, v in values if t >= cutoff and np.isfinite(v)]
        if not window:
            return float("nan"), float("nan")
        arr_t = np.asarray([item[0] for item in window], dtype=float)
        arr_v = np.asarray([item[1] for item in window], dtype=float)
        mean = float(np.nanmean(arr_v))
        slope = float("nan")
        if arr_v.size >= 3 and np.nanmax(arr_t) > np.nanmin(arr_t):
            x = arr_t - arr_t[0]
            slope = float(np.polyfit(x, arr_v, 1)[0])
        return mean, slope

    def _format_value(self, value, unit, slope=False):
        if value is None or not np.isfinite(value):
            return "--"
        if unit == "db":
            text = f"{value:+.2f} dB"
        elif unit == "sigma":
            text = f"{value:+.2f} σ"
        elif unit == "unit":
            text = f"{value:+.3f}"
        elif unit == "score":
            text = f"{value:.1f}x"
        elif unit == "ms":
            text = f"{value:+.3f} ms"
        else:
            text = f"{value:+.3g}"
        return f"{text}/s" if slope else text

    def _effective_tolerance(self, row):
        base = self._parse_float(row["tolerance"].get(), row["spec"]["default_tolerance"])
        scale = self._parse_float(self.tolerance_scale_var.get(), 1.0)
        return max(abs(base * scale), 1e-12)

    def _light_state(self, spec, value, tolerance, baseline_ready):
        if spec.get("needs_baseline", True) and not baseline_ready:
            return "gray", "請先設定基準"
        if not np.isfinite(value):
            return "gray", "尚無資料"
        if spec["mode"] == "high_only":
            if value > tolerance:
                return "red", "高於容忍"
            return "gray", "接近正常"
        if value > tolerance:
            return "red", "高於正常"
        if value < -tolerance:
            return "green", "低於正常"
        return "gray", "接近正常"

    def _update_row(self, key, now=None, append=False):
        row_list = self.rows[key]
        row = row_list[0]
        spec = row["spec"]
        now = time.time() if now is None else float(now)
        value = self._feature_value(spec)
        if append:
            self._append_history(key, now, value)
        short_sec = max(self._parse_float(self.short_window_var.get(), 5.0), 0.1)
        long_sec = max(self._parse_float(self.long_window_var.get(), 30.0), short_sec)
        short_mean, _short_slope = self._window_stats(key, now, short_sec)
        long_mean, long_slope = self._window_stats(key, now, long_sec)
        judge_value = short_mean if np.isfinite(short_mean) else value
        tolerance = self._effective_tolerance(row)
        baseline_ready = self._baseline_ready()
        color, status = self._light_state(spec, judge_value, tolerance, baseline_ready)
        if np.isfinite(long_mean) and np.isfinite(long_slope):
            slope_floor = tolerance / max(long_sec, 1.0) * 0.5
            if long_mean > tolerance and long_slope > slope_floor:
                status = "穩定上升"
                color = "red"
            elif spec["mode"] != "high_only" and long_mean < -tolerance and long_slope < -slope_floor:
                status = "穩定下降"
                color = "green"
        for item in row_list:
            item["current"].set(self._format_value(value, spec["unit"]))
            item["short"].set(self._format_value(short_mean, spec["unit"]))
            item["long"].set(self._format_value(long_mean, spec["unit"]))
            item["slope"].set(self._format_value(long_slope, spec["unit"], slope=True))
            item["status"].set(status)
            item["status_label"].configure(foreground=self.LIGHT_COLORS[color])
            item["light"].itemconfig(item["oval"], fill=self.LIGHT_COLORS[color])

    def _refresh_all_rows(self):
        now = time.time()
        for key in self.rows:
            self._update_row(key, now=now, append=False)

    def apply_calibration_config(self, config):
        """套用外部校準 JSON 中可對應到即時燈號頁的容忍值。"""
        self.calibration_config = config or {}
        self.calibration_name = self.calibration_config.get("created_at", "unknown")
        overrides = self.calibration_config.get("gui_overrides", []) or []
        applied = 0
        for override in overrides:
            key = f"{override.get('source')}.{override.get('metric')}"
            if key not in self.rows:
                continue
            tolerance = override.get("tolerance")
            try:
                tolerance = float(tolerance)
            except Exception:
                continue
            for row in self.rows[key]:
                row["tolerance"].set(self._format_tolerance(max(abs(tolerance), 1e-9)))
            applied += 1
        self._refresh_all_rows()
        selected_count = len(self.calibration_config.get("selected_features", []) or [])
        if applied:
            self.status_var.set(f"已載入校準設定：{selected_count} 個特徵，已套用 {applied} 個即時燈號容忍值。")
        else:
            self.status_var.set(f"已載入校準設定：{selected_count} 個特徵；此版即時燈號沒有直接可套用的 dB 特徵對應。")

    def _poll(self):
        now = time.time()
        for key in self.rows:
            self._update_row(key, now=now, append=True)
        if self.simulation_enabled:
            running = False
            baseline_text = "模擬基準"
            state_text = "燈號模擬中"
        else:
            running = bool(getattr(self.spectrum_page, "running", False))
            baseline_text = "已設定基準" if self._baseline_ready() else "尚未設定基準"
            state_text = "擷取中" if running else "未擷取"
        calibration_text = f"；校準 {self.calibration_name}" if self.calibration_config else ""
        self.status_var.set(f"{state_text}；{baseline_text}{calibration_text}；燈號以短時平均判斷，歷史保留最近 {self.history_seconds:.0f} 秒。")

    def _schedule_poll(self):
        self._poll()
        self.parent.after(500, self._schedule_poll)


class TdmsOfflineRecognitionPage:
    """用正常 TDMS 建 robust baseline，再離線辨識測試 TDMS。"""

    FS_FALLBACK = 1_000_000.0
    EPS = 1e-12
    BANDS = [
        ("35_45k", 35_000.0, 45_000.0),
        ("40_100k", 40_000.0, 100_000.0),
        ("100_150k", 100_000.0, 150_000.0),
        ("150_200k", 150_000.0, 200_000.0),
        ("200_250k", 200_000.0, 250_000.0),
        ("250_300k", 250_000.0, 300_000.0),
        ("100_300k", 100_000.0, 300_000.0),
    ]
    LEGACY_BANDS = [
        ("b00_5_10k", 5_000.0, 10_000.0),
        ("b01_10_15k", 10_000.0, 15_000.0),
        ("b02_15_20k", 15_000.0, 20_000.0),
        ("b03_20_70k", 20_000.0, 70_000.0),
        ("b04_70_120k", 70_000.0, 120_000.0),
        ("b05_120_170k", 120_000.0, 170_000.0),
        ("b06_170_250k", 170_000.0, 250_000.0),
        ("b07_250_350k", 250_000.0, 350_000.0),
        ("b08_350_450k", 350_000.0, 450_000.0),
        ("b09_420_470k", 420_000.0, 470_000.0),
    ]
    CORE_FEATURES = [
        "ai1_40_100k_stft_mean_db",
        "ai1_40_100k_log_energy",
        "diff_40_100k_log_energy",
        "dual_coh_40_100k",
        "dual_cpsd_40_100k_db",
        "dual_gcc_peak_score",
        "rms_ratio_ai1_ai0",
        "rms_ratio_diff_ai1",
    ]

    def __init__(self, parent, spectrum_page=None, feature_lights_page=None):
        self.parent = parent
        self.spectrum_page = spectrum_page
        self.feature_lights_page = feature_lights_page
        self.normal_paths = []
        self.test_path = None
        self.baseline_stats = {}
        self.normal_rows = []
        self.test_rows = []
        self.test_z_rows = []
        self.summary = {}
        self.worker_running = False
        self.calibration_config = None
        self.calibration_path = None

        self.window_sec_var = tk.StringVar(value="0.5")
        self.hop_sec_var = tk.StringVar(value="0.25")
        self.normal_files_var = tk.StringVar(value="尚未選擇 normal TDMS")
        self.test_file_var = tk.StringVar(value="尚未選擇 test TDMS")
        self.calibration_var = tk.StringVar(value="校準設定：未載入")
        self.status_var = tk.StringVar(value="請先選擇 normal TDMS 建立基準。")
        self.result_var = tk.StringVar(value="結果：--")

        self._build_ui(parent)

    def _build_ui(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(3, weight=2)

        controls = ttk.LabelFrame(parent, text="TDMS 離線辨識控制", padding=6)
        controls.grid(row=0, column=0, sticky="ew", padx=6, pady=5)
        ttk.Button(controls, text="選擇正常 TDMS", command=self.select_normal_files).grid(row=0, column=0, padx=3, pady=2)
        ttk.Button(controls, text="建立正常基準", command=self.build_baseline).grid(row=0, column=1, padx=3, pady=2)
        ttk.Button(controls, text="選擇測試 TDMS", command=self.select_test_file).grid(row=0, column=2, padx=3, pady=2)
        ttk.Button(controls, text="開始辨識", command=self.recognize_test_file).grid(row=0, column=3, padx=3, pady=2)
        ttk.Button(controls, text="送到燈號頁", command=self.send_to_feature_lights).grid(row=0, column=4, padx=3, pady=2)
        ttk.Button(controls, text="匯出報告", command=self.export_report).grid(row=0, column=5, padx=3, pady=2)
        ttk.Button(controls, text="載入校準設定", command=self.load_calibration_file).grid(row=0, column=6, padx=3, pady=2)
        ttk.Label(controls, text="window(s)").grid(row=0, column=7, padx=(12, 2), sticky="e")
        ttk.Entry(controls, textvariable=self.window_sec_var, width=6).grid(row=0, column=8, padx=2)
        ttk.Label(controls, text="hop(s)").grid(row=0, column=9, padx=(6, 2), sticky="e")
        ttk.Entry(controls, textvariable=self.hop_sec_var, width=6).grid(row=0, column=10, padx=2)
        ttk.Label(controls, textvariable=self.normal_files_var, foreground="gray").grid(row=1, column=0, columnspan=5, sticky="w", padx=3)
        ttk.Label(controls, textvariable=self.test_file_var, foreground="gray").grid(row=1, column=5, columnspan=3, sticky="w", padx=3)
        ttk.Label(controls, textvariable=self.calibration_var, foreground="#6b4c00").grid(row=1, column=8, columnspan=3, sticky="w", padx=3)

        status = ttk.LabelFrame(parent, text="辨識摘要", padding=6)
        status.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 5))
        ttk.Label(status, textvariable=self.result_var, font=("Arial", 11, "bold"), foreground="#1f4e79").grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.status_var, foreground="blue").grid(row=1, column=0, sticky="w", pady=(3, 0))

        table_frame = ttk.LabelFrame(parent, text="Top abnormal features", padding=5)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 5))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("rank", "feature", "score", "median_z", "max_z")
        self.feature_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        for col, title, width in [
            ("rank", "#", 42),
            ("feature", "特徵", 280),
            ("score", "平均異常", 90),
            ("median_z", "median z", 90),
            ("max_z", "max z", 90),
        ]:
            self.feature_tree.heading(col, text=title)
            self.feature_tree.column(col, width=width, anchor="w")
        self.feature_tree.grid(row=0, column=0, sticky="nsew")
        feature_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.feature_tree.yview)
        feature_scrollbar.grid(row=0, column=1, sticky="ns")
        self.feature_tree.configure(yscrollcommand=feature_scrollbar.set)

        fig_frame = ttk.LabelFrame(parent, text="Window score timeline / Robust-z heatmap", padding=4)
        fig_frame.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))
        fig_frame.columnconfigure(0, weight=1)
        fig_frame.rowconfigure(0, weight=1)
        self.fig = Figure(figsize=(11.5, 4.8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, fig_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._init_plots()

    def _init_plots(self):
        self.fig.clear()
        ax1, ax2 = self.fig.subplots(2, 1, height_ratios=[1, 1.4])
        ax1.set_title("Window leak score")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("score")
        ax1.grid(alpha=0.2)
        ax2.set_title("Top feature robust-z heatmap")
        ax2.set_xlabel("window")
        ax2.set_ylabel("feature")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _parse_config(self):
        window_sec = max(float(self.window_sec_var.get()), 0.05)
        hop_sec = max(float(self.hop_sec_var.get()), 0.01)
        if hop_sec > window_sec:
            hop_sec = window_sec
            self.hop_sec_var.set(f"{hop_sec:g}")
        return window_sec, hop_sec

    def select_normal_files(self):
        paths = filedialog.askopenfilenames(
            title="選擇 normal TDMS 檔案",
            filetypes=[("TDMS files", "*.tdms"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.normal_paths = [Path(p) for p in paths]
        self.normal_files_var.set(f"Normal: {len(self.normal_paths)} 檔")
        self.status_var.set("已選擇 normal TDMS，請建立正常基準。")

    def select_test_file(self):
        path = filedialog.askopenfilename(
            title="選擇測試 TDMS 檔案",
            filetypes=[("TDMS files", "*.tdms"), ("All files", "*.*")],
        )
        if not path:
            return
        self.test_path = Path(path)
        self.test_file_var.set(f"Test: {self.test_path.name}")
        self.status_var.set("已選擇測試 TDMS，可開始辨識。")

    def load_calibration_file(self):
        default_path = Path(r"C:\Users\m143340\Desktop\M304_2ch_0505\analysis_m304a_0508\outputs\auto_threshold_calibration\m304a_leak_calibration_latest.json")
        initial_dir = default_path.parent if default_path.exists() else Path(r"C:\Users\m143340\Desktop\M304_2ch_0505\analysis_m304a_0508\outputs")
        path = filedialog.askopenfilename(
            title="載入 M304A 校準 JSON",
            initialdir=str(initial_dir),
            filetypes=[("Calibration JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
            if config.get("schema_version") != "m304a_leak_threshold_calibration_v1":
                raise ValueError("JSON schema_version 不是 m304a_leak_threshold_calibration_v1")
            selected = config.get("selected_features", [])
            baselines = config.get("distance_baselines", {})
            if not selected or not baselines:
                raise ValueError("JSON 缺少 selected_features 或 distance_baselines")
            self.calibration_config = config
            self.calibration_path = Path(path)
            created_at = config.get("created_at", "unknown")
            self.calibration_var.set(f"校準設定：{created_at} / {len(selected)} 特徵")
            if self.feature_lights_page is not None and hasattr(self.feature_lights_page, "apply_calibration_config"):
                self.feature_lights_page.apply_calibration_config(config)
            self.status_var.set(f"已載入校準設定：{self.calibration_path.name}")
        except Exception as exc:
            self.calibration_config = None
            self.calibration_path = None
            messagebox.showerror("載入校準設定失敗", f"{exc}")
            self.status_var.set("校準設定載入失敗，已保留原本 robust baseline 流程。")

    def _set_status(self, text):
        self.parent.after(0, lambda: self.status_var.set(text))

    def build_baseline(self):
        if self.worker_running:
            messagebox.showinfo("TDMS 離線辨識", "目前已有分析工作執行中。")
            return
        if not self.normal_paths:
            messagebox.showwarning("TDMS 離線辨識", "請先選擇 normal TDMS 檔案。")
            return
        try:
            window_sec, hop_sec = self._parse_config()
        except Exception as exc:
            messagebox.showerror("參數錯誤", str(exc))
            return
        self.worker_running = True
        self.status_var.set("建立正常基準中...")
        thread = threading.Thread(target=self._build_baseline_worker, args=(list(self.normal_paths), window_sec, hop_sec), daemon=True)
        thread.start()

    def _build_baseline_worker(self, paths, window_sec, hop_sec):
        try:
            rows = []
            for idx, path in enumerate(paths, start=1):
                self._set_status(f"建立正常基準中... {idx}/{len(paths)}: {path.name}")
                rows.extend(self._extract_file_features(path, window_sec, hop_sec))
            stats = self._make_baseline_stats(rows)
            self.parent.after(0, lambda: self._finish_baseline(rows, stats))
        except Exception as exc:
            ErrorHandler.log_error(exc, "offline tdms baseline")
            self.parent.after(0, lambda e=exc: self._worker_failed(f"建立正常基準失敗：{e}"))

    def _finish_baseline(self, rows, stats):
        self.worker_running = False
        self.normal_rows = rows
        self.baseline_stats = stats
        warn = "；基準樣本偏少，建議至少 10-20 windows" if len(rows) < 10 else ""
        self.status_var.set(f"正常基準已建立：{len(self.normal_paths)} 檔 / {len(rows)} windows / {len(stats)} 特徵{warn}")
        self.result_var.set("結果：正常基準就緒")

    def recognize_test_file(self):
        if self.worker_running:
            messagebox.showinfo("TDMS 離線辨識", "目前已有分析工作執行中。")
            return
        if not self.baseline_stats and not self.calibration_config:
            messagebox.showwarning("TDMS 離線辨識", "請先建立正常基準，或載入校準設定 JSON。")
            return
        if self.test_path is None:
            messagebox.showwarning("TDMS 離線辨識", "請先選擇測試 TDMS。")
            return
        try:
            window_sec, hop_sec = self._parse_config()
        except Exception as exc:
            messagebox.showerror("參數錯誤", str(exc))
            return
        self.worker_running = True
        self.status_var.set("測試檔辨識中...")
        thread = threading.Thread(target=self._recognize_worker, args=(self.test_path, window_sec, hop_sec), daemon=True)
        thread.start()

    def _recognize_worker(self, path, window_sec, hop_sec):
        try:
            rows = self._extract_file_features(path, window_sec, hop_sec)
            if self.calibration_config:
                z_rows, summary = self._recognize_with_calibration(path, rows)
            else:
                z_rows = self._robust_z_rows(rows)
                summary = self._summarize_result(rows, z_rows)
            self.parent.after(0, lambda: self._finish_recognition(rows, z_rows, summary))
        except Exception as exc:
            ErrorHandler.log_error(exc, "offline tdms recognition")
            self.parent.after(0, lambda e=exc: self._worker_failed(f"測試檔辨識失敗：{e}"))

    def _parse_distance_from_path(self, path):
        match = re.search(r"straight(\d+)cm", Path(path).name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _recognize_with_calibration(self, path, rows):
        config = self.calibration_config or {}
        distance = self._parse_distance_from_path(path)
        baselines = config.get("distance_baselines", {})
        distance_stats = baselines.get(str(distance)) if distance is not None else None
        if not distance_stats:
            available = ", ".join(sorted(baselines.keys()))
            raise ValueError(f"校準 JSON 沒有 {distance} cm 的 baseline；可用距離：{available}")

        selected = config.get("selected_features", []) or []
        score_rule = config.get("score_rule", {}) or {}
        mode_name = score_rule.get("default_mode", "balanced")
        active_rule = (score_rule.get("modes", {}) or {}).get(mode_name, {})
        top_n = max(1, int(active_rule.get("top_n_features", len(selected) or 1)))
        score_threshold = float(active_rule.get("score_threshold", 1.0))
        active_selected = selected[:top_n]
        z_rows = []
        window_scores = []
        suspicious_flags = []
        top_feature_rows = []
        for row in rows:
            z = {"window_index": row["window_index"], "window_start_sec": row["window_start_sec"]}
            weighted_scores = []
            warning_hits = 0
            strong_hits = 0
            hit_details = []
            for spec in active_selected:
                feature = spec.get("feature_name")
                stat = distance_stats.get(feature)
                value = row.get(feature, float("nan"))
                if not feature or not stat or not np.isfinite(value):
                    continue
                robust_z = float((value - stat["median"]) / max(stat["iqr"], self.EPS))
                z[feature] = robust_z
                direction = spec.get("direction", "higher_is_leak")
                score = self._signed_calibrated_score(robust_z, direction)
                warning = float(spec.get("warning_threshold_score", 1.0))
                strong = float(spec.get("strong_threshold_score", max(1.5, warning * 1.8)))
                weight = float(spec.get("weight", 1.0))
                weighted_scores.append(score / max(warning, self.EPS) * weight)
                if score > warning:
                    warning_hits += 1
                    hit_details.append((feature, score, robust_z))
                if score > strong:
                    strong_hits += 1
            leak_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0
            suspicious = bool(leak_score >= score_threshold)
            z["_calibrated_leak_score"] = leak_score
            z["_warning_hits"] = warning_hits
            z["_strong_hits"] = strong_hits
            z_rows.append(z)
            window_scores.append(leak_score)
            suspicious_flags.append(suspicious)
            top_feature_rows.extend(hit_details)

        suspicious_ratio = float(np.mean(suspicious_flags)) if suspicious_flags else 0.0
        max_score = float(np.max(window_scores)) if window_scores else 0.0
        max_idx = int(np.argmax(window_scores)) if window_scores else 0
        if suspicious_ratio >= 0.2 or max_score >= score_threshold:
            label = "疑似洩漏"
        elif suspicious_ratio > 0.05 or max_score >= score_threshold * 0.75:
            label = "不確定"
        else:
            label = "接近正常"

        feature_summary = {}
        for feature, score, robust_z in top_feature_rows:
            stats_row = feature_summary.setdefault(feature, {"score": [], "z": []})
            stats_row["score"].append(score)
            stats_row["z"].append(robust_z)
        top_features = []
        for feature, values in feature_summary.items():
            top_features.append({
                "feature": feature,
                "mean_score": float(np.mean(values["score"])),
                "median_z": float(np.median(values["z"])),
                "max_z": float(np.max(values["z"])),
            })
        top_features = sorted(top_features, key=lambda item: item["mean_score"], reverse=True)[:20]
        return z_rows, {
            "label": label,
            "overall_leak_score": float(np.mean(window_scores)) if window_scores else 0.0,
            "suspicious_window_ratio": suspicious_ratio,
            "max_window_score": max_score,
            "window_scores": window_scores,
            "suspicious_flags": suspicious_flags,
            "most_suspicious_time_sec": float(z_rows[max_idx].get("window_start_sec", 0.0)) if z_rows else 0.0,
            "top_features": top_features,
            "calibration_path": str(self.calibration_path) if self.calibration_path else "",
            "calibration_mode": mode_name,
            "score_threshold": score_threshold,
        }

    def _signed_calibrated_score(self, robust_z, direction):
        if direction == "lower_is_leak":
            return -float(robust_z)
        if direction == "two_sided":
            return abs(float(robust_z))
        return float(robust_z)

    def _finish_recognition(self, rows, z_rows, summary):
        self.worker_running = False
        self.test_rows = rows
        self.test_z_rows = z_rows
        self.summary = summary
        self.result_var.set(
            f"結果：{summary['label']} | score={summary['overall_leak_score']:.2f} | "
            f"可疑窗比例={summary['suspicious_window_ratio']:.1%} | max={summary['max_window_score']:.2f}"
        )
        self.status_var.set(f"辨識完成：最可疑時間約 {summary['most_suspicious_time_sec']:.3f} 秒。")
        self._update_feature_table(summary["top_features"])
        self._update_plots(summary)

    def _worker_failed(self, message):
        self.worker_running = False
        self.status_var.set(message)
        messagebox.showerror("TDMS 離線辨識", message)

    def _read_tdms_pair(self, path):
        if not TDMS_AVAILABLE or TdmsFile is None:
            raise RuntimeError("nptdms 未安裝，無法讀取 TDMS。")
        tdms = TdmsFile.read(str(path))
        group = tdms.groups()[0]
        channels = {ch.name: ch for ch in group.channels()}
        if "ai0" in channels and "ai1" in channels:
            ch0, ch1 = channels["ai0"], channels["ai1"]
        else:
            ch_list = group.channels()
            if len(ch_list) < 2:
                raise ValueError(f"{path.name} 不包含雙通道資料。")
            ch0, ch1 = ch_list[0], ch_list[1]
        ai0 = np.asarray(ch0[:], dtype=np.float32)
        ai1 = np.asarray(ch1[:], dtype=np.float32)
        n = min(ai0.size, ai1.size)
        inc = getattr(ch0, "properties", {}).get("wf_increment")
        fs = self.FS_FALLBACK
        if inc:
            try:
                fs = 1.0 / float(inc)
            except Exception:
                fs = self.FS_FALLBACK
        return ai0[:n], ai1[:n], float(fs)

    def _window_starts(self, n, fs, window_sec, hop_sec):
        window_size = int(round(fs * window_sec))
        hop_size = int(round(fs * hop_sec))
        window_size = max(2048, min(window_size, n))
        hop_size = max(1, min(hop_size, window_size))
        if n < window_size:
            return np.asarray([], dtype=np.int64), window_size, hop_size
        return np.arange(0, n - window_size + 1, hop_size, dtype=np.int64), window_size, hop_size

    def _extract_file_features(self, path, window_sec, hop_sec):
        ai0, ai1, fs = self._read_tdms_pair(path)
        starts, window_size, _hop_size = self._window_starts(min(ai0.size, ai1.size), fs, window_sec, hop_sec)
        if starts.size == 0:
            raise ValueError(f"{path.name}: 檔案太短，無法切出 window。")
        rows = []
        for idx, start in enumerate(starts):
            end = start + window_size
            x0 = self._demean(ai0[start:end])
            x1 = self._demean(ai1[start:end])
            diff = (x1 - x0).astype(np.float32)
            row = {
                "filename": path.name,
                "filepath": str(path),
                "window_index": int(idx),
                "window_start_sec": float(start / fs),
                "sample_rate_hz": float(fs),
            }
            for prefix, x in [("ai0", x0), ("ai1", x1), ("diff", diff)]:
                row.update(self._time_features(x, prefix))
                row.update(self._frequency_features(x, fs, prefix))
                row.update(self._stft_features(x, fs, prefix))
            row["rms_ratio_ai1_ai0"] = row["ai1_rms"] / (row["ai0_rms"] + self.EPS)
            row["rms_ratio_diff_ai1"] = row["diff_rms"] / (row["ai1_rms"] + self.EPS)
            row["rms_ratio_diff_ai0"] = row["diff_rms"] / (row["ai0_rms"] + self.EPS)
            row["energy_delta_ai1_minus_ai0"] = row["ai1_rms"] ** 2 - row["ai0_rms"] ** 2
            for band_name, _low, _high in self.LEGACY_BANDS:
                ai0_energy = row.get(f"{band_name}_ai0_energy", 0.0)
                ai1_energy = row.get(f"{band_name}_ai1_energy", 0.0)
                diff_energy = row.get(f"{band_name}_diff_energy", 0.0)
                row[f"{band_name}_ratio_ai1_ai0"] = ai1_energy / (ai0_energy + self.EPS)
                row[f"{band_name}_ratio_diff_ai1"] = diff_energy / (ai1_energy + self.EPS)
            row.update(self._dual_features(x0, x1, fs))
            rows.append(row)
        return rows

    def _demean(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        return (x - float(np.mean(x))).astype(np.float32)

    def _time_features(self, x, prefix):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return {}
        rms = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        std = float(np.std(x))
        centered = x - float(np.mean(x))
        if std > 0:
            skewness = float(np.mean((centered / std) ** 3))
            kurt = float(np.mean((centered / std) ** 4))
        else:
            skewness = 0.0
            kurt = 0.0
        signs = np.signbit(centered)
        zcr = float(np.mean(signs[1:] != signs[:-1])) if centered.size > 1 else 0.0
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std": std,
            f"{prefix}_rms": rms,
            f"{prefix}_peak": peak,
            f"{prefix}_ptp": float(np.ptp(x)),
            f"{prefix}_crest": float(peak / (rms + self.EPS)),
            f"{prefix}_kurtosis": kurt,
            f"{prefix}_skewness": skewness,
            f"{prefix}_zcr": zcr,
        }

    def _frequency_features(self, x, fs, prefix):
        x = np.asarray(x, dtype=np.float64)
        nperseg = min(4096, max(256, x.size))
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")
        use = (freqs >= 5_000.0) & (freqs <= min(300_000.0, fs * 0.49))
        f_use = freqs[use]
        p_use = psd[use] + self.EPS
        total = float(np.sum(p_use))
        if f_use.size and total > 0:
            dom_freq = float(f_use[int(np.argmax(p_use))])
            centroid = float(np.sum(f_use * p_use) / total)
            prob = p_use / total
            entropy = float(-np.sum(prob * np.log2(prob + self.EPS)))
            flatness = float(np.exp(np.mean(np.log(p_use))) / (np.mean(p_use) + self.EPS))
        else:
            dom_freq = centroid = entropy = flatness = 0.0
        out = {
            f"{prefix}_dom_freq_hz": dom_freq,
            f"{prefix}_centroid_hz": centroid,
            f"{prefix}_entropy": entropy,
            f"{prefix}_flatness": flatness,
        }
        for name, low, high in self.BANDS:
            mask = (freqs >= low) & (freqs < high)
            energy = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            out[f"{prefix}_{name}_energy"] = energy
            out[f"{prefix}_{name}_log_energy"] = float(np.log10(energy + self.EPS))
        for name, low, high in self.LEGACY_BANDS:
            mask = (freqs >= low) & (freqs < high)
            energy = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            out[f"{name}_{prefix}_energy"] = energy
        for low_k in range(5, 500, 5):
            high_k = low_k + 5
            low = low_k * 1000.0
            high = high_k * 1000.0
            mask = (freqs >= low) & (freqs < high)
            energy = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            out[f"scan_{low_k:03d}_{high_k:03d}k_{prefix}_energy"] = energy
        return out

    def _stft_features(self, x, fs, prefix):
        x = np.asarray(x, dtype=np.float64)
        nperseg = min(2048, max(512, x.size // 8))
        nperseg = min(nperseg, x.size)
        if nperseg < 256:
            return {}
        freqs, _times, sxx = signal.spectrogram(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend=False,
            scaling="density",
            mode="psd",
        )
        db = 10.0 * np.log10(sxx + 1e-30)
        out = {}
        for name, low, high in self.BANDS:
            mask = (freqs >= low) & (freqs < high)
            values = db[mask, :]
            out[f"{prefix}_{name}_stft_mean_db"] = float(np.nanmean(values)) if values.size else float("nan")
            out[f"{prefix}_{name}_stft_max_db"] = float(np.nanmax(values)) if values.size else float("nan")
        return out

    def _band_mean_1d(self, freqs_hz, values, low, high):
        mask = (freqs_hz >= low) & (freqs_hz < high) & np.isfinite(values)
        return float(np.nanmean(values[mask])) if np.any(mask) else float("nan")

    def _dual_features(self, ai0, ai1, fs):
        out = {}
        n = min(ai0.size, ai1.size)
        if n < 2048:
            return out
        x = np.asarray(ai0[:n], dtype=np.float64)
        y = np.asarray(ai1[:n], dtype=np.float64)
        nperseg = min(8192, max(1024, n // 8))
        nperseg = min(nperseg, n)
        freqs, coh = signal.coherence(x, y, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend=False)
        csd_freqs, pxy = signal.csd(x, y, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend=False, scaling="density")
        cpsd_db = 10.0 * np.log10(np.abs(pxy) + 1e-30)
        for name, low, high in self.BANDS:
            out[f"dual_coh_{name}"] = self._band_mean_1d(freqs, coh, low, high)
            out[f"dual_cpsd_{name}_db"] = self._band_mean_1d(csd_freqs, cpsd_db, low, high)
        step = max(1, int(fs // 100_000))
        x_small = x[::step]
        y_small = y[::step]
        if x_small.size > 16 and y_small.size > 16:
            corr = signal.correlate(y_small, x_small, mode="full", method="fft")
            denom = np.sqrt(np.sum(x_small * x_small)) * np.sqrt(np.sum(y_small * y_small)) + self.EPS
            center = corr.size // 2
            max_lag = min(int(0.002 * fs / step), center)
            segment = corr[center - max_lag:center + max_lag + 1]
            idx = int(np.argmax(np.abs(segment)))
            lag_samples = (idx - max_lag) * step
            out["xcorr_ai1_ai0_max"] = float(segment[idx] / denom)
            out["xcorr_ai1_ai0_lag_samples"] = float(lag_samples)
            out["xcorr_ai1_ai0_lag_us"] = float(lag_samples / fs * 1_000_000.0)
        out.update(self._gcc_phat_features(x, y, fs))
        return out

    def _gcc_phat_features(self, ai0, ai1, fs, low_hz=40_000.0, high_hz=100_000.0):
        n = min(ai0.size, ai1.size)
        if n < 2048:
            return {"dual_gcc_peak_score": float("nan"), "dual_gcc_delay_ms": float("nan")}
        x = np.asarray(ai0[:n], dtype=np.float64) - float(np.mean(ai0[:n]))
        y = np.asarray(ai1[:n], dtype=np.float64) - float(np.mean(ai1[:n]))
        nfft = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
        x_fft = np.fft.rfft(x, n=nfft)
        y_fft = np.fft.rfft(y, n=nfft)
        cross = y_fft * np.conj(x_fft)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / max(float(fs), 1.0))
        band = (freqs >= low_hz) & (freqs <= min(high_hz, fs * 0.49))
        if np.any(band):
            cross[~band] = 0.0
        cross = cross / (np.abs(cross) + 1e-20)
        corr = np.fft.irfft(cross, n=nfft)
        max_shift = min(int(0.005 * fs), nfft // 2 - 1)
        corr = np.concatenate([corr[-max_shift:], corr[:max_shift + 1]])
        lags = np.arange(-max_shift, max_shift + 1, dtype=float) / fs
        abs_corr = np.abs(corr)
        idx = int(np.nanargmax(abs_corr))
        score = float(abs_corr[idx] / (np.nanmedian(abs_corr) + 1e-20))
        return {"dual_gcc_peak_score": score, "dual_gcc_delay_ms": float(lags[idx] * 1000.0)}

    def _numeric_feature_names(self, rows):
        excluded = {"filename", "filepath", "window_index", "window_start_sec", "sample_rate_hz"}
        names = sorted({key for row in rows for key, value in row.items() if key not in excluded and isinstance(value, (int, float, np.floating))})
        return names

    def _make_baseline_stats(self, rows):
        stats = {}
        for name in self._numeric_feature_names(rows):
            values = np.asarray([row.get(name, np.nan) for row in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            p10, p25, med, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
            iqr = max(float(p75 - p25), self.EPS)
            stats[name] = {"median": float(med), "iqr": iqr, "p10": float(p10), "p90": float(p90), "n": int(values.size)}
        return stats

    def _robust_z_rows(self, rows):
        z_rows = []
        for row in rows:
            z = {"window_index": row["window_index"], "window_start_sec": row["window_start_sec"]}
            for name, stat in self.baseline_stats.items():
                value = row.get(name, float("nan"))
                z[name] = float((value - stat["median"]) / max(stat["iqr"], self.EPS)) if np.isfinite(value) else float("nan")
            z_rows.append(z)
        return z_rows

    def _summarize_result(self, rows, z_rows):
        feature_names = [name for name in self.baseline_stats if name not in {"window_index", "window_start_sec"}]
        positive_scores = []
        window_scores = []
        suspicious_flags = []
        for z in z_rows:
            vals = np.asarray([max(0.0, z.get(name, 0.0)) for name in feature_names if np.isfinite(z.get(name, float("nan")))], dtype=float)
            core_vals = np.asarray([max(0.0, z.get(name, 0.0)) for name in self.CORE_FEATURES if np.isfinite(z.get(name, float("nan")))], dtype=float)
            score = float(np.nanmean(np.sort(vals)[-5:])) if vals.size else 0.0
            core_hits = int(np.sum(core_vals > 3.0)) if core_vals.size else 0
            suspicious = bool(core_hits >= 2 or score >= 5.0)
            window_scores.append(score)
            suspicious_flags.append(suspicious)
        suspicious_ratio = float(np.mean(suspicious_flags)) if suspicious_flags else 0.0
        max_score = float(np.max(window_scores)) if window_scores else 0.0
        max_idx = int(np.argmax(window_scores)) if window_scores else 0
        if suspicious_ratio >= 0.2 or max_score >= 5.0:
            label = "疑似洩漏"
        elif suspicious_ratio > 0.05 or max_score >= 3.0:
            label = "不確定"
        else:
            label = "接近正常"
        top_features = self._top_abnormal_features(z_rows, feature_names)
        return {
            "label": label,
            "overall_leak_score": float(np.mean(window_scores)) if window_scores else 0.0,
            "suspicious_window_ratio": suspicious_ratio,
            "max_window_score": max_score,
            "window_scores": window_scores,
            "suspicious_flags": suspicious_flags,
            "most_suspicious_time_sec": float(z_rows[max_idx].get("window_start_sec", 0.0)) if z_rows else 0.0,
            "top_features": top_features,
        }

    def _top_abnormal_features(self, z_rows, feature_names, top_k=20):
        rows = []
        for name in feature_names:
            vals = np.asarray([z.get(name, np.nan) for z in z_rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            positive = np.maximum(vals, 0.0)
            rows.append({
                "feature": name,
                "score": float(np.nanmean(positive)),
                "median_z": float(np.nanmedian(vals)),
                "max_z": float(np.nanmax(vals)),
            })
        rows.sort(key=lambda item: (item["score"], item["max_z"]), reverse=True)
        return rows[:top_k]

    def _update_feature_table(self, top_features):
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        for idx, row in enumerate(top_features, start=1):
            self.feature_tree.insert("", "end", values=(idx, row["feature"], f"{row['score']:.2f}", f"{row['median_z']:.2f}", f"{row['max_z']:.2f}"))

    def _update_plots(self, summary):
        self.fig.clear()
        ax1, ax2 = self.fig.subplots(2, 1, height_ratios=[1, 1.4])
        times = np.asarray([row.get("window_start_sec", 0.0) for row in self.test_z_rows], dtype=float)
        scores = np.asarray(summary.get("window_scores", []), dtype=float)
        ax1.plot(times[: scores.size], scores, color="#c0392b", linewidth=1.8)
        ax1.axhline(3.0, color="#d35400", linestyle="--", linewidth=0.9)
        ax1.axhline(5.0, color="#c0392b", linestyle="--", linewidth=0.9)
        ax1.set_title("Window leak score")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("score")
        ax1.grid(alpha=0.2)
        top = summary.get("top_features", [])[:12]
        if top and self.test_z_rows:
            feature_names = [item["feature"] for item in top]
            mat = np.asarray([[row.get(name, np.nan) for row in self.test_z_rows] for name in feature_names], dtype=float)
            mat = np.clip(mat, -6, 6)
            im = ax2.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-6, vmax=6, origin="lower")
            ax2.set_yticks(np.arange(len(feature_names)))
            ax2.set_yticklabels(feature_names, fontsize=8)
            ax2.set_xlabel("window")
            ax2.set_title("Top feature robust-z heatmap")
            self.fig.colorbar(im, ax=ax2, label="robust z")
        else:
            ax2.text(0.5, 0.5, "尚無 heatmap", ha="center", va="center", transform=ax2.transAxes, color="gray")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def send_to_feature_lights(self):
        if not self.test_rows:
            messagebox.showwarning("送到燈號頁", "請先完成測試檔辨識。")
            return
        if self.spectrum_page is None:
            messagebox.showwarning("送到燈號頁", "找不到即時 Spectrum 頁。")
            return
        aggregate = self._aggregate_test_features()
        self.spectrum_page.latest_metrics = {"ai0": aggregate["ai0"], "ai1": aggregate["ai1"]}
        self.spectrum_page.latest_dual_metrics = aggregate["dual"]
        # 讓「即時特徵燈號」把離線 normal baseline 視為已建立，
        # 才能用離線測試檔摘要直接更新紅/綠/灰燈號。
        self.spectrum_page.baseline = {
            "ai0": {"source": "tdms_offline"},
            "ai1": {"source": "tdms_offline"},
        }
        self.spectrum_page.dual_baseline = {"source": "tdms_offline"}
        if self.feature_lights_page is not None:
            self.feature_lights_page.simulation_enabled = False
            self.feature_lights_page._clear_history()
        self.status_var.set("已將離線測試摘要送到即時特徵燈號頁。")

    def _feature_delta(self, feature_name):
        values = np.asarray([row.get(feature_name, np.nan) for row in self.test_rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        stat = self.baseline_stats.get(feature_name)
        if not stat:
            return float(np.nanmedian(values))
        return float(np.nanmedian(values) - stat["median"])

    def _aggregate_test_features(self):
        def channel_metrics(prefix):
            return {
                "has_delta": True,
                "delta_peak_freq_khz": float("nan"),
                "delta_peak_time_sec": float("nan"),
                "delta_peak_db": self._feature_delta(f"{prefix}_40_100k_stft_max_db"),
                "delta_band_35_45_db": self._feature_delta(f"{prefix}_35_45k_stft_mean_db"),
                "delta_band_40_100_db": self._feature_delta(f"{prefix}_40_100k_stft_mean_db"),
                "delta_band_40_100_max_db": self._feature_delta(f"{prefix}_40_100k_stft_max_db"),
                "delta_band_100_300_db": self._feature_delta(f"{prefix}_100_300k_stft_mean_db"),
                "delta_band_250_300_db": self._feature_delta(f"{prefix}_250_300k_stft_mean_db"),
            }
        dual = {
            "has_dual": True,
            "has_dual_delta": True,
            "coh_delta_35_45": self._feature_delta("dual_coh_35_45k"),
            "coh_delta_40_100": self._feature_delta("dual_coh_40_100k"),
            "coh_delta_100_300": self._feature_delta("dual_coh_100_300k"),
            "coh_delta_250_300": self._feature_delta("dual_coh_250_300k"),
            "cpsd_delta_40_100_db": self._feature_delta("dual_cpsd_40_100k_db"),
            "gcc_peak_score": float(np.nanmedian([row.get("dual_gcc_peak_score", np.nan) for row in self.test_rows])),
            "gcc_delay_ms": float(np.nanmedian([row.get("dual_gcc_delay_ms", np.nan) for row in self.test_rows])),
        }
        return {"ai0": channel_metrics("ai0"), "ai1": channel_metrics("ai1"), "dual": dual}

    def export_report(self):
        if not self.summary:
            messagebox.showwarning("匯出報告", "請先完成測試檔辨識。")
            return
        out_dir = filedialog.askdirectory(title="選擇報告輸出資料夾")
        if not out_dir:
            return
        out_dir = Path(out_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"tdms_offline_features_{stamp}.csv"
        md_path = out_dir / f"tdms_offline_report_{stamp}.md"
        png_path = out_dir / f"tdms_offline_figures_{stamp}.png"
        self._write_rows_csv(csv_path, self.test_rows, self.test_z_rows)
        self.fig.savefig(png_path, dpi=160)
        self._write_report_md(md_path, csv_path.name, png_path.name)
        self.status_var.set(f"已匯出報告：{md_path}")

    def _write_rows_csv(self, path, rows, z_rows):
        fieldnames = []
        merged_rows = []
        for row, z in zip(rows, z_rows):
            merged = dict(row)
            for key, value in z.items():
                if key not in {"window_index", "window_start_sec"}:
                    merged[f"z_{key}"] = value
            merged_rows.append(merged)
            for key in merged:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

    def _write_report_md(self, path, csv_name, png_name):
        top_lines = "\n".join(
            f"- {item['feature']}: score={item['score']:.2f}, median_z={item['median_z']:.2f}, max_z={item['max_z']:.2f}"
            for item in self.summary.get("top_features", [])[:10]
        )
        text = f"""# TDMS 離線辨識報告

- 結果：{self.summary.get('label', '--')}
- overall_leak_score：{self.summary.get('overall_leak_score', 0.0):.3f}
- suspicious_window_ratio：{self.summary.get('suspicious_window_ratio', 0.0):.1%}
- max_window_score：{self.summary.get('max_window_score', 0.0):.3f}
- most_suspicious_time_sec：{self.summary.get('most_suspicious_time_sec', 0.0):.3f}
- normal_files：{len(self.normal_paths)}
- normal_windows：{len(self.normal_rows)}
- test_file：{self.test_path.name if self.test_path else '--'}
- features_csv：{csv_name}
- figures：{png_name}

## Top abnormal features
{top_lines}
"""
        path.write_text(text, encoding="utf-8")

class DataCollectionGUI:
    """數據收集系統主介面 v17"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("數據收集系統 v17 - 真同步雙通道 transition 3ch 輸出版")
        
        # 設定視窗大小
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        if screen_width >= 1920:
            window_width = 1600
            window_height = 1000
        elif screen_width >= 1366:
            window_width = 1400
            window_height = 900
        else:
            window_width = 1200
            window_height = 800
        
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1000, 700)
        
        # 多通道擷取器
        self.multi_capture = MultiChannelCapture()
        
        # 收集狀態
        self.is_collecting = False
        self.collection_count = 0
        
        # 建構GUI
        self._build_gui()
        self._reset_standard_session()
    
    def _build_gui(self):
        """建構主介面"""
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True)
        
        # 頁面1：標準數據收集
        self.tab_standard = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_standard, text="標準數據")
        self._build_standard_page(self.tab_standard)

        # 頁面2：雙感測器即時 AE spectrogram
        self.tab_realtime_spectrum = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_realtime_spectrum, text="即時 AE Spectrum")
        self.realtime_spectrum_page = RealtimeAESpectrumPage(self.tab_realtime_spectrum)

        # 頁面3：即時特徵值與燈號監看，讀取 Spectrum 頁已計算好的特徵
        self.tab_feature_lights = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_feature_lights, text="即時特徵燈號")
        self.realtime_feature_lights_page = RealtimeFeatureLightsPage(self.tab_feature_lights, self.realtime_spectrum_page)

        # 頁面4：TDMS 離線 normal baseline 與洩漏辨識，不啟動 DAQ reader。
        self.tab_tdms_offline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tdms_offline, text="TDMS 離線辨識")
        self.tdms_offline_page = TdmsOfflineRecognitionPage(
            self.tab_tdms_offline,
            self.realtime_spectrum_page,
            self.realtime_feature_lights_page,
        )
        
        # GAN 噪音頁已移除；Standard 頁同時負責雙通道同步與舊 near/far 流程。
    
    # ==================== 標準數據收集頁面 ====================
    
    def _build_standard_page(self, parent):
        """建構標準數據收集頁面"""
        
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)
        
        # 左側使用Canvas+Scrollbar實現滾動
        left_container = ttk.Frame(parent)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        
        # 創建Canvas和Scrollbar
        canvas = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        left_panel = ttk.Frame(canvas)
        
        # 配置滾動
        left_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=left_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 佈局Canvas和Scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 滑鼠滾輪支援
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        right_panel = ttk.Frame(parent)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        
        # ===== 左側:設定區 =====
        
        # 設備配置 (壓縮版)
        lf_device = ttk.LabelFrame(left_panel, text="設備", padding=3)
        lf_device.pack(fill="x", pady=1)
        
        # 設備配置 (壓縮版)
        lf_device = ttk.LabelFrame(left_panel, text="設備", padding=3)
        lf_device.pack(fill="x", pady=1)
        
        # 設備名稱和模擬模式在同一行
        dev_row = ttk.Frame(lf_device)
        dev_row.pack(fill="x", pady=1)
        ttk.Label(dev_row, text="設備:", width=5).pack(side="left")
        self.std_device_name = ttk.Entry(dev_row, width=12)
        self.std_device_name.insert(0, "cDAQ1Mod1")
        self.std_device_name.pack(side="left", padx=2)
        
        self.std_simulation_mode = tk.BooleanVar(value=not NIDAQ_AVAILABLE)
        ttk.Checkbutton(dev_row, text="模擬", 
                       variable=self.std_simulation_mode,
                       command=self._update_std_device_mode).pack(side="left", padx=2)
        
        # 狀態（更緊湊）
        self.std_device_status = ttk.Label(lf_device, text="", 
                                           font=('Arial', 7), foreground="gray")
        self.std_device_status.pack(pady=1)
        
        # 檔案命名（縮減版）
        lf_file = ttk.LabelFrame(left_panel, text="檔案", padding=3)
        lf_file.pack(fill="x", pady=1)
        
        # 檔案命名（縮減版）
        lf_file = ttk.LabelFrame(left_panel, text="檔案", padding=3)
        lf_file.pack(fill="x", pady=1)
        
        folder_frame = ttk.Frame(lf_file)
        folder_frame.pack(fill="x", pady=1)
        ttk.Label(folder_frame, text="路徑:", width=5).pack(side="left")
        self.std_folder = ttk.Entry(folder_frame, width=12)
        self.std_folder.insert(0, "./leak_dataset")
        self.std_folder.pack(side="left", fill="x", expand=True, padx=1)
        ttk.Button(folder_frame, text="...", width=2,
                  command=self._browse_standard_folder).pack(side="right")
        self.std_folder.bind('<FocusOut>', lambda e: self._prepare_standard_folder())
        self.std_folder.bind('<Return>', lambda e: self._prepare_standard_folder())
        self.std_folder_status = ttk.Label(lf_file, text="選定條件母資料夾後會自動建立 dual_2ch 與舊 near/far",
                                           font=('Arial', 7), foreground="gray", wraplength=220)
        self.std_folder_status.pack(fill="x", pady=1)
        
        # 洩漏有無
        leak_frame = ttk.Frame(lf_file)
        leak_frame.pack(fill="x", pady=1)
        self.std_leak_status = tk.StringVar(value="leak1")
        ttk.Radiobutton(leak_frame, text="leak1", variable=self.std_leak_status, 
                       value="leak1", command=self._update_std_filename).pack(side="left", padx=3)
        ttk.Radiobutton(leak_frame, text="leak0", variable=self.std_leak_status, 
                       value="leak0", command=self._update_std_filename).pack(side="left")
        
        # 參數（緊湊排列）
        params_grid = ttk.Frame(lf_file)
        params_grid.pack(fill="x", pady=1)
        
        ttk.Label(params_grid, text="洩漏量:", width=8).grid(row=0, column=0, sticky="w")
        self.std_volume = ttk.Entry(params_grid, width=8)
        self.std_volume.insert(0, "400")
        self.std_volume.grid(row=0, column=1, sticky="w", padx=2)
        ttk.Label(params_grid, text="ml").grid(row=0, column=2, sticky="w")
        
        ttk.Label(params_grid, text="壓力:", width=8).grid(row=1, column=0, sticky="w", pady=2)
        self.std_pressure = ttk.Entry(params_grid, width=8)
        self.std_pressure.insert(0, "3")
        self.std_pressure.grid(row=1, column=1, sticky="w", padx=2)
        ttk.Label(params_grid, text="bar").grid(row=1, column=2, sticky="w")
        
        ttk.Label(params_grid, text="距離:", width=8).grid(row=2, column=0, sticky="w", pady=2)
        pos_frame = ttk.Frame(params_grid)
        pos_frame.grid(row=2, column=1, columnspan=2, sticky="ew")
        self.std_position_type = ttk.Combobox(pos_frame, width=8, state="readonly")
        self.std_position_type['values'] = ("straight", "left", "right")
        self.std_position_type.current(0)
        self.std_position_type.pack(side="left", padx=2)
        self.std_position_dist = ttk.Entry(pos_frame, width=5)
        self.std_position_dist.insert(0, "250")
        self.std_position_dist.pack(side="left", padx=2)
        ttk.Label(pos_frame, text="cm").pack(side="left")
        
        # 綁定更新
        for widget in [self.std_volume, self.std_pressure, self.std_position_dist]:
            widget.bind('<KeyRelease>', lambda e: self._update_std_filename())
        self.std_position_type.bind('<<ComboboxSelected>>', lambda e: self._update_std_filename())
        
        # 檔名預覽（更緊湊）
        ttk.Separator(lf_file, orient='horizontal').pack(fill="x", pady=2)
        self.std_filename_preview = ttk.Label(lf_file, text="", foreground="blue",
                                             font=('Courier', 7), wraplength=200)
        self.std_filename_preview.pack(fill="x", pady=1)
        
        # 擷取參數
        lf_params = ttk.LabelFrame(left_panel, text="參數", padding=3)
        lf_params.pack(fill="x", pady=1)
        
        params_grid2 = ttk.Frame(lf_params)
        params_grid2.pack(fill="x")
        
        ttk.Label(params_grid2, text="採樣率:", width=8).grid(row=0, column=0, sticky="w")
        self.std_sample_rate = ttk.Entry(params_grid2, width=10)
        self.std_sample_rate.insert(0, "1000000")
        self.std_sample_rate.grid(row=0, column=1, sticky="w", padx=1)
        ttk.Label(params_grid2, text="Hz", font=('Arial', 7)).grid(row=0, column=2, sticky="w")
        
        ttk.Label(params_grid2, text="時間:", width=8).grid(row=1, column=0, sticky="w", pady=1)
        self.std_duration = ttk.Entry(params_grid2, width=10)
        self.std_duration.insert(0, "0.5")
        self.std_duration.grid(row=1, column=1, sticky="w", padx=1, pady=1)
        ttk.Label(params_grid2, text="秒", font=('Arial', 7)).grid(row=1, column=2, sticky="w", pady=1)
        
        ttk.Label(params_grid2, text="點數:", width=8).grid(row=2, column=0, sticky="w", pady=1)
        self.std_sample_num = ttk.Entry(params_grid2, width=10)
        self.std_sample_num.insert(0, "500000")
        self.std_sample_num.grid(row=2, column=1, sticky="w", padx=1, pady=1)
        
        self.std_sample_rate.bind('<KeyRelease>', lambda e: self._update_std_params('rate'))
        self.std_duration.bind('<KeyRelease>', lambda e: self._update_std_params('time'))
        self.std_sample_num.bind('<KeyRelease>', lambda e: self._update_std_params('points'))
        
        ttk.Label(lf_params, text="時間=點數/採樣率", 
                 font=('Arial', 7), foreground="gray").pack(pady=1)

        # Standard 擷取模式
        lf_acq = ttk.LabelFrame(left_panel, text="擷取模式", padding=3)
        lf_acq.pack(fill="x", pady=1)
        self.std_acquisition_mode = tk.StringVar(value="dual")
        ttk.Radiobutton(
            lf_acq,
            text="雙通道同步 ai0 / ai1",
            variable=self.std_acquisition_mode,
            value="dual",
            command=self._update_standard_acquisition_mode,
        ).pack(anchor="w", padx=5, pady=1)
        ttk.Radiobutton(
            lf_acq,
            text="單通道舊模式 near/far",
            variable=self.std_acquisition_mode,
            value="legacy",
            command=self._update_standard_acquisition_mode,
        ).pack(anchor="w", padx=5, pady=1)
        
        # 通道配置
        lf_channels = ttk.LabelFrame(left_panel, text="通道", padding=3)
        lf_channels.pack(fill="x", pady=1)
        
        # 表頭
        header_frame = ttk.Frame(lf_channels)
        header_frame.pack(fill="x", pady=1)
        ttk.Label(header_frame, text="啟用", width=4, font=('Arial', 7)).grid(row=0, column=0)
        ttk.Label(header_frame, text="名稱", width=8, font=('Arial', 7)).grid(row=0, column=1)
        
        # 通道列表（更緊湊）
        self.std_channels = {}
        self.std_channel_names = {}
        for i, default_ch in enumerate(['ai0', 'ai1', 'ai2', 'ai3']):
            ch_row = ttk.Frame(lf_channels)
            ch_row.pack(fill="x", pady=0)
            
            var = tk.BooleanVar(value=(i < 2))
            cb = ttk.Checkbutton(ch_row, variable=var)
            cb.grid(row=0, column=0, padx=5)
            
            entry = ttk.Entry(ch_row, width=10, font=('Arial', 8))
            entry.insert(0, default_ch)
            entry.grid(row=0, column=1, padx=1)
            
            self.std_channels[default_ch] = var
            self.std_channel_names[default_ch] = entry
        
        # 單通道採集角色
        lf_role = ttk.LabelFrame(left_panel, text="near/far 角色", padding=3)
        lf_role.pack(fill="x", pady=1)

        self.std_sensor_role = tk.StringVar(value="near")
        role_row = ttk.Frame(lf_role)
        role_row.pack(fill="x", pady=1)
        self.std_role_buttons = []
        rb_near = ttk.Radiobutton(role_row, text="near", variable=self.std_sensor_role,
                                  value="near")
        rb_near.pack(side="left", padx=5)
        rb_far = ttk.Radiobutton(role_row, text="far", variable=self.std_sensor_role,
                                 value="far")
        rb_far.pack(side="left", padx=5)
        self.std_role_buttons.extend([rb_near, rb_far])

        self.std_session_status = ttk.Label(
            lf_role,
            text="near: 0 筆 | far: 0 筆 | 尚未可輸出",
            font=('Arial', 8),
            foreground="gray",
            wraplength=230,
        )
        self.std_session_status.pack(fill="x", pady=1)

        ttk.Button(lf_role, text="重新整理暫存數量",
                  command=self._refresh_standard_staging_status).pack(fill="x", padx=5, pady=2)

        ttk.Button(lf_role, text="清空本次暫存",
                  command=self._reset_standard_session).pack(fill="x", padx=5, pady=2)

        # Transition 標記：用於 normal -> leak / leak -> normal 的單通道資料。
        self.std_transition_enabled = tk.BooleanVar(value=False)
        self.std_transition_state = tk.StringVar(value="normal")
        self.std_transition_status = ttk.Label(
            lf_role,
            text="transition: 未啟用",
            font=('Arial', 8),
            foreground="gray",
            wraplength=230,
        )
        self.std_transition_status.pack(fill="x", pady=(4, 1))
        self.std_transition_widgets = []
        self.std_transition_check = ttk.Checkbutton(
            lf_role,
            text="啟用 normal/leak 切換標記",
            variable=self.std_transition_enabled,
            command=self._toggle_transition_mode,
        )
        self.std_transition_check.pack(fill="x", padx=5, pady=1)
        self.std_transition_widgets.append(self.std_transition_check)

        transition_btn_row = ttk.Frame(lf_role)
        transition_btn_row.pack(fill="x", padx=5, pady=1)
        self.btn_std_mark_normal = ttk.Button(transition_btn_row, text="標記 normal",
                   command=lambda: self._mark_transition_state("normal"))
        self.btn_std_mark_normal.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.btn_std_mark_leak = ttk.Button(transition_btn_row, text="標記 leak",
                   command=lambda: self._mark_transition_state("leak"))
        self.btn_std_mark_leak.pack(side="left", fill="x", expand=True, padx=(2, 0))
        self.btn_std_clear_transition = ttk.Button(lf_role, text="清除本次標記",
                  command=self._clear_transition_events)
        self.btn_std_clear_transition.pack(fill="x", padx=5, pady=1)
        self.std_transition_widgets.extend([
            self.btn_std_mark_normal,
            self.btn_std_mark_leak,
            self.btn_std_clear_transition,
        ])
        self.btn_std_transition_output = tk.Button(
            lf_role,
            text="輸出 transition 3ch",
            command=self._std_output_transition_session,
            state='disabled',
            font=('Arial', 8),
            height=1,
            cursor="hand2",
        )
        self.btn_std_transition_output.pack(fill="x", padx=5, pady=2)
        self.std_transition_widgets.append(self.btn_std_transition_output)
        ttk.Label(
            lf_role,
            text="建議: interval 1-2秒；near優先；50/100/250/500ml × 0/20/50/70cm × 3 repeats",
            font=('Arial', 7),
            foreground="gray",
            wraplength=230,
        ).pack(fill="x", padx=5, pady=(1, 3))
        
        # Disturbance 標記：用於 hard-negative 干擾事件標註
        self.std_disturbance_status = ttk.Label(
            lf_role,
            text="disturbance: 0 events",
            font=('Arial', 8),
            foreground="gray",
            wraplength=230,
        )
        self.std_disturbance_status.pack(fill="x", pady=(6, 1))
        
        self.std_disturbance_type = tk.StringVar(value="tap_pipe")
        self.std_disturbance_strength = tk.StringVar(value="medium")
        
        disturbance_type_row = ttk.Frame(lf_role)
        disturbance_type_row.pack(fill="x", padx=5, pady=1)
        ttk.Label(disturbance_type_row, text="干擾類型", width=8).pack(side="left")
        ttk.Combobox(
            disturbance_type_row,
            textvariable=self.std_disturbance_type,
            state="readonly",
            values=("tap_pipe", "touch_pipe", "cable_pull", "adjacent_vibration", "other"),
            width=16,
        ).pack(side="left", fill="x", expand=True)
        
        disturbance_strength_row = ttk.Frame(lf_role)
        disturbance_strength_row.pack(fill="x", padx=5, pady=1)
        ttk.Label(disturbance_strength_row, text="強度", width=8).pack(side="left")
        ttk.Combobox(
            disturbance_strength_row,
            textvariable=self.std_disturbance_strength,
            state="readonly",
            values=("light", "medium", "strong"),
            width=16,
        ).pack(side="left", fill="x", expand=True)
        
        disturbance_btn_row = ttk.Frame(lf_role)
        disturbance_btn_row.pack(fill="x", padx=5, pady=1)
        self.btn_std_disturbance_start = ttk.Button(
            disturbance_btn_row,
            text="干擾開始",
            command=self._mark_disturbance_start,
        )
        self.btn_std_disturbance_start.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        self.btn_std_disturbance_end = ttk.Button(
            disturbance_btn_row,
            text="干擾結束",
            command=self._mark_disturbance_end,
        )
        self.btn_std_disturbance_end.pack(side="left", fill="x", expand=True, padx=(2, 0))
        
        self.btn_std_clear_disturbance = ttk.Button(
            lf_role,
            text="清除干擾標記",
            command=self._clear_disturbance_events,
        )
        self.btn_std_clear_disturbance.pack(fill="x", padx=5, pady=1)

        # 模式選擇
        lf_mode = ttk.LabelFrame(left_panel, text="模式", padding=3)
        lf_mode.pack(fill="both", expand=True, pady=1)
        
        mode_nb = ttk.Notebook(lf_mode)
        mode_nb.pack(fill="both", expand=True)
        
        # 單次模式
        single_frame = ttk.Frame(mode_nb, padding=2)
        mode_nb.add(single_frame, text="單次")
        
        self.btn_std_single_capture = tk.Button(single_frame, text="擷取訊號",
                                                command=self._std_single_capture,
                                                bg="#4CAF50", fg="white",
                                                font=('Arial', 9, 'bold'),
                                                height=1, cursor="hand2")
        self.btn_std_single_capture.pack(fill="x", padx=5, pady=2)
        
        self.btn_std_single_save = tk.Button(single_frame, text="輸出 dual_2ch",
                                            command=self._std_output_session,
                                            state='disabled',
                                            font=('Arial', 8),
                                            height=1, cursor="hand2")
        self.btn_std_single_save.pack(fill="x", padx=5, pady=2)
        
        # 長時間模式
        long_frame = ttk.Frame(mode_nb, padding=2)
        mode_nb.add(long_frame, text="長時間")
        
        # 說明文字
        info_label = ttk.Label(long_frame, 
                              text="自動連續擷取並儲存\n(使用上方採樣參數)",
                              font=('Arial', 7), foreground="gray")
        info_label.pack(pady=1)
        
        # 緊湊排列
        grid_long = ttk.Frame(long_frame)
        grid_long.pack(fill="x", padx=3, pady=2)
        
        ttk.Label(grid_long, text="總時長:", width=6, font=('Arial', 8)).grid(row=0, column=0, sticky="w")
        self.std_long_duration = ttk.Entry(grid_long, width=5, font=('Arial', 8))
        self.std_long_duration.insert(0, "10")
        self.std_long_duration.grid(row=0, column=1)
        self.std_long_unit = ttk.Combobox(grid_long, width=4, state="readonly", font=('Arial', 8))
        self.std_long_unit['values'] = ("秒", "分", "時")
        self.std_long_unit.current(1)
        self.std_long_unit.grid(row=0, column=2, padx=1)
        
        ttk.Label(grid_long, text="間隔:", width=6, font=('Arial', 8)).grid(row=1, column=0, sticky="w", pady=1)
        self.std_long_interval = ttk.Entry(grid_long, width=5, font=('Arial', 8))
        self.std_long_interval.insert(0, "0.5")
        self.std_long_interval.grid(row=1, column=1, pady=1)
        self.std_long_interval_unit = ttk.Combobox(grid_long, width=4, state="readonly", font=('Arial', 8))
        self.std_long_interval_unit['values'] = ("秒", "分")
        self.std_long_interval_unit.current(0)
        self.std_long_interval_unit.grid(row=1, column=2, padx=1, pady=1)
        
        # 預估和說明
        estimate_frame = ttk.Frame(long_frame)
        estimate_frame.pack(fill="x", pady=1)
        
        self.std_estimate = ttk.Label(estimate_frame, text="預估: -- 筆", 
                                      foreground="blue", font=('Arial', 7))
        self.std_estimate.pack()
        
        ttk.Label(estimate_frame, text="每筆自動儲存", 
                 font=('Arial', 7), foreground="gray").pack()
        
        # 顯示選項
        self.std_show_last = tk.BooleanVar(value=True)
        ttk.Checkbutton(long_frame, text="顯示最後一筆", 
                       variable=self.std_show_last,
                       style='Small.TCheckbutton').pack(pady=1)
        
        self.btn_std_long_start = tk.Button(long_frame, text="開始",
                                            command=self._std_long_start,
                                            bg="#4CAF50", fg="white",
                                            font=('Arial', 9, 'bold'),
                                            height=1, cursor="hand2")
        self.btn_std_long_start.pack(fill="x", padx=5, pady=2)
        
        self.btn_std_long_stop = tk.Button(long_frame, text="停止",
                                           command=self._std_long_stop,
                                           bg="#f44336", fg="white",
                                           font=('Arial', 9, 'bold'),
                                           height=1, cursor="hand2",
                                           state='disabled')
        self.btn_std_long_stop.pack(fill="x", padx=5, pady=2)

        self.btn_std_long_output = tk.Button(long_frame, text="輸出 dual_2ch",
                                             command=self._std_output_session,
                                             state='disabled',
                                             font=('Arial', 8),
                                             height=1, cursor="hand2")
        self.btn_std_long_output.pack(fill="x", padx=5, pady=2)
        
        for widget in [self.std_long_duration, self.std_long_interval]:
            widget.bind('<KeyRelease>', lambda e: self._update_std_estimate())
        for widget in [self.std_long_unit, self.std_long_interval_unit]:
            widget.bind('<<ComboboxSelected>>', lambda e: self._update_std_estimate())
        
        # ===== 右側:顯示區 =====
        
        # 狀態
        self.std_status = ttk.Label(right_panel, text="就緒", 
                                    font=('Arial', 9), foreground="gray")
        self.std_status.pack(pady=2)
        
        # Notebook for different views
        display_nb = ttk.Notebook(right_panel)
        display_nb.pack(fill="both", expand=True)
        
        # Tab 1: 頻寬波形顯示
        waveform_frame = ttk.Frame(display_nb)
        display_nb.add(waveform_frame, text="頻寬波形")
        self.waveform_viewer = WaveformViewer(waveform_frame)
        
        # Tab 2: 進度追蹤
        progress_frame = ttk.Frame(display_nb)
        display_nb.add(progress_frame, text="進度")
        self.progress_tracker = ProgressTracker(progress_frame)
        
        # 初始化
        self._update_std_filename()
        self._update_std_estimate()
        self._update_std_device_mode()
        self._prepare_standard_folder()
        self._update_standard_acquisition_mode()

    def _browse_standard_folder(self):
        """瀏覽並準備標準資料條件母資料夾。"""
        self._browse_folder(self.std_folder)
        self._prepare_standard_folder()

    def _prepare_standard_folder(self):
        """建立 near/far/merged_3ch/_session_staging 子資料夾，並嘗試套用條件資訊。"""
        try:
            folder = Path(self.std_folder.get()).expanduser()
            dirs = ConditionFolderHelper.ensure_session_dirs(folder)
            parsed = ConditionFolderHelper.parse(folder)
            if parsed:
                self._apply_condition_folder_meta(parsed)
                self.std_folder_status.config(
                    text=f"已準備: ai0 / ai1 / dual_2ch / transition_dual / 舊 near/far | 條件: {folder.name}",
                    foreground="green",
                )
            else:
                self.std_folder_status.config(
                    text="已準備子資料夾；資料夾名稱無法自動解析，將使用左側欄位產生檔名",
                    foreground="orange",
                )
            return dirs
        except Exception as e:
            self.std_folder_status.config(text=f"資料夾準備失敗: {str(e)[:60]}", foreground="red")
            return None

    def _apply_condition_folder_meta(self, meta):
        """從 50_leak-near 0 cm / normal-near 0 cm 套用 GUI 欄位。"""
        self.std_leak_status.set(meta["leak_status"])

        self.std_volume.delete(0, tk.END)
        self.std_volume.insert(0, str(meta["volume_ml"]))

        self.std_position_type.set(meta["position_type"])
        self.std_position_dist.delete(0, tk.END)
        self.std_position_dist.insert(0, str(meta["distance_cm"]))

        self._update_std_filename()

    def _reset_standard_session(self):
        """清空目前 GUI session 的 near/far 暫存索引與暫存檔。"""
        old_session_id = getattr(self, "std_session_id", None)
        if old_session_id:
            try:
                staging_root = Path(self.std_folder.get()).expanduser() / "_session_staging"
                shutil.rmtree(staging_root / old_session_id)
            except Exception:
                pass
        self.std_session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.std_staged_records = {"near": [], "far": [], "dual": []}
        self.std_session_lock = None
        self.std_transition_events = []
        self.std_transition_start_time = None
        self.std_transition_initial_state = "normal"
        self.std_session_start_time = None
        self.std_timeline_start_time = None  # 保留給舊程式相容
        if hasattr(self, "std_transition_state"):
            self.std_transition_state.set("normal")
        self._update_transition_status()
        self._update_standard_session_status()
        
        # 加入干擾
        self.std_disturbance_events = []
        self.std_active_disturbance = None
        self.std_disturbance_start_time = None
        self._update_disturbance_status()
        

    def _standard_counts(self):
        records = getattr(self, "std_staged_records", {"near": [], "far": [], "dual": []})
        return len(records.get("near", [])), len(records.get("far", []))

    def _dual_count(self):
        records = getattr(self, "std_staged_records", {"dual": []})
        return len(records.get("dual", []))

    def _is_dual_standard_mode(self):
        mode = getattr(self, "std_acquisition_mode", None)
        if mode is None:
            return True
        try:
            return mode.get() == "dual"
        except Exception:
            return str(mode) == "dual"

    def _update_standard_acquisition_mode(self):
        dual_mode = self._is_dual_standard_mode()
        if dual_mode and hasattr(self, "std_channels"):
            for default_ch, var in self.std_channels.items():
                var.set(default_ch in ("ai0", "ai1"))

        role_state = 'disabled' if dual_mode else 'normal'
        for widget in getattr(self, "std_role_buttons", []):
            try:
                widget.config(state=role_state)
            except Exception:
                pass

        transition_state = 'normal'
        for widget in getattr(self, "std_transition_widgets", []):
            try:
                widget.config(state=transition_state)
            except Exception:
                pass

        button_text = "輸出 dual_2ch" if dual_mode else "輸出 near/far + merged"
        transition_button_text = "輸出 transition 3ch" if dual_mode else "輸出 transition single-channel"
        for btn_name in ("btn_std_single_save", "btn_std_long_output"):
            if hasattr(self, btn_name):
                getattr(self, btn_name).config(text=button_text)
        if hasattr(self, "btn_std_transition_output"):
            self.btn_std_transition_output.config(text=transition_button_text)

        if hasattr(self, "std_status"):
            if dual_mode:
                self.std_status.config(text="雙通道同步模式：ai0、ai1", foreground="blue")
            else:
                self.std_status.config(text="單通道舊模式：使用 near/far 角色暫存再合併", foreground="blue")
        self._update_transition_status()
        self._update_standard_session_status()

    def _transition_enabled(self):
        var = getattr(self, "std_transition_enabled", None)
        if var is None:
            return False
        try:
            return bool(var.get())
        except Exception:
            return bool(var)

    def _transition_total_count(self):
        if self._is_dual_standard_mode():
            return self._dual_count()
        near_count, far_count = self._standard_counts()
        return near_count + far_count

    def _transition_now_elapsed(self):
        if getattr(self, "std_session_start_time", None) is None:
            now = time.time()
            self.std_session_start_time = now
            self.std_timeline_start_time = now
            self.std_transition_start_time = now  # 只保留顯示相容
            self.std_transition_initial_state = (
                self.std_transition_state.get() if hasattr(self, "std_transition_state") else "normal"
            )
        return max(0.0, time.time() - self.std_session_start_time)

    def _transition_state_at_elapsed(self, elapsed_sec):
        state = getattr(self, "std_transition_initial_state", "normal")
        for event in sorted(getattr(self, "std_transition_events", []), key=lambda item: float(item["event_elapsed_sec"])):
            if float(event["event_elapsed_sec"]) <= float(elapsed_sec):
                state = event["to_state"]
            else:
                break
        return state

    def _transition_segment_info(self, start_elapsed, end_elapsed):
        mid_elapsed = (float(start_elapsed) + float(end_elapsed)) / 2.0
        events = getattr(self, "std_transition_events", [])
        events_in_window = [
            event for event in events
            if float(start_elapsed) <= float(event["event_elapsed_sec"]) <= float(end_elapsed)
        ]
        nearest_offset = ""
        if events:
            nearest_offset = min(
                abs(mid_elapsed - float(event["event_elapsed_sec"]))
                for event in events
            )
        state = self._transition_state_at_elapsed(mid_elapsed)
        segment_type = "transition" if events_in_window else "steady"
        return {
            "capture_start_elapsed_sec": float(start_elapsed),
            "capture_end_elapsed_sec": float(end_elapsed),
            "capture_mid_elapsed_sec": float(mid_elapsed),
            "state_label": state,
            "file_label": 1 if state == "leak" else 0,
            "segment_type": segment_type,
            "nearest_event_offset_sec": nearest_offset,
        }

    def _toggle_transition_mode(self):
        if self._transition_enabled():
            if self._transition_total_count() > 0:
                self.std_status.config(text="已啟用 transition；若已有舊暫存，建議先清空本次暫存", foreground="orange")
            else:
                self.std_status.config(text="已啟用 transition 標記模式", foreground="blue")
            self._transition_now_elapsed()
        else:
            self.std_status.config(text="已停用 transition 標記模式", foreground="gray")
        self._update_transition_status()
        self._update_standard_session_status()

    def _update_transition_status(self):
        if not hasattr(self, "std_transition_status"):
            return
        enabled = self._transition_enabled()
        state = self.std_transition_state.get() if hasattr(self, "std_transition_state") else "normal"
        event_count = len(getattr(self, "std_transition_events", []))
        elapsed = 0.0
        if getattr(self, "std_session_start_time", None) is not None:
            elapsed = time.time() - self.std_session_start_time
        if enabled:
            text = f"transition: ON | 目前 {state} | events {event_count} | {elapsed:.1f}s"
            color = "blue" if state == "normal" else "red"
        else:
            text = f"transition: OFF | 目前 {state} | events {event_count}"
            color = "gray"
        self.std_transition_status.config(text=text, foreground=color)

    def _mark_transition_state(self, new_state):
        if new_state not in ("normal", "leak"):
            return
        if not self._transition_enabled():
            self.std_status.config(text="請先啟用 transition 標記模式", foreground="orange")
            return

        old_state = self.std_transition_state.get()
        elapsed = self._transition_now_elapsed()
        event = {
            "event_index": len(getattr(self, "std_transition_events", [])) + 1,
            "event_wall_time": datetime.now().isoformat(timespec="seconds"),
            "event_elapsed_sec": float(elapsed),
            "from_state": old_state,
            "to_state": new_state,
            "collection_count_at_event": int(getattr(self, "collection_count", 0)),
        }
        self.std_transition_events.append(event)
        self.std_transition_state.set(new_state)
        self._update_transition_status()
        self.std_status.config(
            text=f"已標記 transition event #{event['event_index']}: {old_state} -> {new_state} @ {elapsed:.1f}s",
            foreground="blue" if new_state == "normal" else "red",
        )

    def _clear_transition_events(self):
        self.std_transition_events = []
        self.std_transition_initial_state = (
            self.std_transition_state.get() if hasattr(self, "std_transition_state") else "normal"
        )
        if self._transition_enabled():
            self._standard_timeline_now_elapsed()
        self._update_transition_status()
        self.std_status.config(text="已清除本次 transition 標記", foreground="blue")

    def _update_standard_session_status(self):
        if not hasattr(self, "std_session_status"):
            return
        if self._is_dual_standard_mode():
            dual_count = self._dual_count()
            transition_mode = self._transition_enabled()
            if transition_mode:
                if dual_count > 0:
                    text = f"transition dual: {dual_count} 筆 | 可輸出 3ch with diff"
                    color = "blue"
                else:
                    text = "transition dual: 0 筆 | ai0/ai1 擷取後可輸出"
                    color = "gray"
                stable_state = 'disabled'
                transition_state = 'normal' if dual_count > 0 and not self.is_collecting else 'disabled'
            else:
                if dual_count > 0:
                    text = f"雙通道同步: {dual_count} 筆 | 可輸出 dual_2ch"
                    color = "green"
                else:
                    text = "雙通道同步: 0 筆 | ai0/ai1 擷取後可輸出"
                    color = "gray"
                stable_state = 'normal' if dual_count > 0 and not self.is_collecting else 'disabled'
                transition_state = 'disabled'
            self.std_session_status.config(text=text, foreground=color)
            for btn_name in ("btn_std_single_save", "btn_std_long_output"):
                if hasattr(self, btn_name):
                    getattr(self, btn_name).config(state=stable_state, text="輸出 dual_2ch")
            if hasattr(self, "btn_std_transition_output"):
                self.btn_std_transition_output.config(state=transition_state, text="輸出 transition 3ch")
            return

        near_count, far_count = self._standard_counts()
        pair_count = min(near_count, far_count)
        transition_mode = self._transition_enabled()
        transition_count = near_count + far_count
        if transition_mode:
            text = f"transition: near {near_count} 筆 | far {far_count} 筆 | 可輸出單通道 {transition_count} 筆"
            color = "blue" if transition_count else "gray"
            self.std_session_status.config(text=text, foreground=color)
            stable_state = 'disabled'
            transition_state = 'normal' if transition_count > 0 and not self.is_collecting else 'disabled'
            for btn_name in ("btn_std_single_save", "btn_std_long_output"):
                if hasattr(self, btn_name):
                    getattr(self, btn_name).config(state=stable_state)
            if hasattr(self, "btn_std_transition_output"):
                self.btn_std_transition_output.config(state=transition_state)
            return

        ready = pair_count > 0
        if near_count == far_count and ready:
            text = f"near: {near_count} 筆 | far: {far_count} 筆 | 可輸出 {pair_count} 組"
            color = "green"
        elif ready:
            extra_role = "near" if near_count > far_count else "far"
            extra_count = abs(near_count - far_count)
            text = f"near: {near_count} 筆 | far: {far_count} 筆 | 可輸出 {pair_count} 組，{extra_role} 多 {extra_count} 筆輸出後刪除"
            color = "orange"
        else:
            text = f"near: {near_count} 筆 | far: {far_count} 筆 | near/far 各至少 1 筆才可輸出"
            color = "orange" if near_count or far_count else "gray"
        self.std_session_status.config(text=text, foreground=color)

        state = 'normal' if ready and not self.is_collecting else 'disabled'
        for btn_name in ("btn_std_single_save", "btn_std_long_output"):
            if hasattr(self, btn_name):
                getattr(self, btn_name).config(state=state)
        if hasattr(self, "btn_std_transition_output"):
            self.btn_std_transition_output.config(state='disabled')
            
    def _disturbance_now_elapsed(self):
        return self._standard_timeline_now_elapsed()
    
    def _standard_timeline_now_elapsed(self):
        if getattr(self, "std_session_start_time", None) is None:
            now = time.time()
            self.std_session_start_time = now
            self.std_timeline_start_time = now
            if getattr(self, "std_transition_start_time", None) is None:
                self.std_transition_start_time = now
        return max(0.0, time.time() - self.std_session_start_time)

    def _update_disturbance_status(self):
        if not hasattr(self, "std_disturbance_status"):
            return
        count = len(getattr(self, "std_disturbance_events", []))
        active = getattr(self, "std_active_disturbance", None)

        if active is not None:
            elapsed = self._disturbance_now_elapsed()
            text = (
                f"disturbance: recording | {active['disturbance_type']} | "
                f"events {count} | {elapsed:.1f}s"
            )
            color = "red"
        else:
            text = f"disturbance: {count} events"
            color = "gray"

        self.std_disturbance_status.config(text=text, foreground=color)

    def _mark_disturbance_start(self):
        if self.std_active_disturbance is not None:
            self.std_status.config(text="目前已有一個干擾事件尚未結束", foreground="orange")
            return

        elapsed = self._disturbance_now_elapsed()
        self.std_active_disturbance = {
            "event_index": len(getattr(self, "std_disturbance_events", [])) + 1,
            "disturbance_type": self.std_disturbance_type.get(),
            "disturbance_strength": self.std_disturbance_strength.get(),
            "start_wall_time": datetime.now().isoformat(timespec="seconds"),
            "start_elapsed_sec": float(elapsed),
            "collection_count_at_event": int(getattr(self, "collection_count", 0)),
        }
        self._update_disturbance_status()
        self.std_status.config(
            text=f"已標記干擾開始: {self.std_active_disturbance['disturbance_type']} @ {elapsed:.1f}s",
            foreground="blue",
        )

    def _mark_disturbance_end(self):
        if self.std_active_disturbance is None:
            self.std_status.config(text="目前沒有進行中的干擾事件", foreground="orange")
            return

        elapsed = self._disturbance_now_elapsed()
        event = dict(self.std_active_disturbance)
        event["end_wall_time"] = datetime.now().isoformat(timespec="seconds")
        event["end_elapsed_sec"] = float(elapsed)
        event["duration_sec"] = max(0.0, event["end_elapsed_sec"] - event["start_elapsed_sec"])

        self.std_disturbance_events.append(event)
        self.std_active_disturbance = None
        self._update_disturbance_status()
        self.std_status.config(
            text=f"已標記干擾結束: {event['disturbance_type']} {event['start_elapsed_sec']:.1f}s ~ {event['end_elapsed_sec']:.1f}s",
            foreground="blue",
        )

    def _clear_disturbance_events(self):
        self.std_disturbance_events = []
        self.std_active_disturbance = None
        self._update_disturbance_status()
        self.std_status.config(text="已清除本次干擾標記", foreground="blue")
    
    # disturbance segment 標註函式
    def _disturbance_segment_info(self, start_elapsed, end_elapsed):
        start_elapsed = float(start_elapsed)
        end_elapsed = float(end_elapsed)

        if end_elapsed < start_elapsed:
            start_elapsed, end_elapsed = end_elapsed, start_elapsed

        window_duration = max(1e-9, end_elapsed - start_elapsed)
        overlaps = []

        for event in getattr(self, "std_disturbance_events", []):
            ev_start = float(event.get("start_elapsed_sec", 0.0) or 0.0)
            ev_end = float(event.get("end_elapsed_sec", ev_start) or ev_start)

            overlap_start = max(start_elapsed, ev_start)
            overlap_end = min(end_elapsed, ev_end)
            overlap_sec = max(0.0, overlap_end - overlap_start)

            if overlap_sec > 0.0:
                overlaps.append((overlap_sec, event))

        active = getattr(self, "std_active_disturbance", None)
        if active is not None:
            ev_start = float(active.get("start_elapsed_sec", 0.0) or 0.0)
            ev_end = end_elapsed
            overlap_start = max(start_elapsed, ev_start)
            overlap_end = min(end_elapsed, ev_end)
            overlap_sec = max(0.0, overlap_end - overlap_start)

            if overlap_sec > 0.0:
                active_event = dict(active)
                overlaps.append((overlap_sec, active_event))

        if not overlaps:
            return {
                "disturbance_present": 0,
                "disturbance_event_count": 0,
                "disturbance_overlap_sec": 0.0,
                "disturbance_overlap_ratio": 0.0,
                "disturbance_primary_type": "",
                "disturbance_primary_strength": "",
            }

        total_overlap = sum(item[0] for item in overlaps)
        primary_overlap, primary_event = max(overlaps, key=lambda item: item[0])

        return {
            "disturbance_present": 1,
            "disturbance_event_count": len(overlaps),
            "disturbance_overlap_sec": float(total_overlap),
            "disturbance_overlap_ratio": float(min(1.0, total_overlap / window_duration)),
            "disturbance_primary_type": primary_event.get("disturbance_type", ""),
            "disturbance_primary_strength": primary_event.get("disturbance_strength", ""),
        }

    def _refresh_standard_staging_status(self):
        """重新掃描目前 session 的硬碟暫存檔，讓手動刪檔後 GUI 狀態同步。"""
        try:
            dirs = ConditionFolderHelper.ensure_session_dirs(Path(self.std_folder.get()).expanduser())
            session_dir = dirs["staging"] / self.std_session_id
            refreshed = {"near": [], "far": [], "dual": []}
            lock_meta = None

            for role in ("near", "far"):
                role_dir = session_dir / role
                if not role_dir.exists():
                    continue
                for path in sorted(role_dir.glob(f"{role}_*.npz")):
                    try:
                        metadata = StageIO.load_metadata(path)
                    except Exception:
                        continue
                    if metadata.get("role") != role:
                        metadata["role"] = role
                    refreshed[role].append({"path": str(path), "metadata": metadata})
                    if lock_meta is None:
                        lock_meta = metadata

            dual_dir = session_dir / "dual"
            if dual_dir.exists():
                for path in sorted(dual_dir.glob("dual_*.npz")):
                    try:
                        metadata = StageIO.load_metadata(path)
                    except Exception:
                        continue
                    metadata["role"] = "dual"
                    refreshed["dual"].append({"path": str(path), "metadata": metadata})
                    if lock_meta is None:
                        lock_meta = metadata

            self.std_staged_records = refreshed
            self.std_session_lock = dict(lock_meta) if lock_meta else None
            near_count, far_count = self._standard_counts()
            dual_count = self._dual_count()
            self._update_standard_session_status()
            self.std_status.config(
                text=f"已重新整理暫存數量: dual {dual_count} 筆、near {near_count} 筆、far {far_count} 筆",
                foreground="blue",
            )
        except Exception as e:
            ErrorHandler.log_error(e, "重新整理暫存數量")
            self.std_status.config(text=f"重新整理失敗: {str(e)[:60]}", foreground="red")

    def _clean_int_text(self, value, default="0"):
        text = str(value).strip().replace("bar", "").replace("ml", "").replace("cm", "")
        try:
            return str(int(float(text)))
        except Exception:
            return default

    def _standard_signature(self):
        leak_status = self.std_leak_status.get()
        volume = "0" if leak_status == "leak0" else self._clean_int_text(self.std_volume.get(), "0")
        pressure = self._clean_int_text(self.std_pressure.get(), "3")
        distance = self._clean_int_text(self.std_position_dist.get(), "0")
        parsed = ConditionFolderHelper.parse(Path(self.std_folder.get()))
        return {
            "condition_folder": Path(self.std_folder.get()).name,
            "leak_status": leak_status,
            "volume_ml": int(volume),
            "matched_volume_ml": int(parsed["matched_volume"]) if parsed else int(volume),
            "pressure_bar": int(pressure),
            "position_type": self.std_position_type.get(),
            "distance_cm": int(distance),
            "sample_rate": int(self.std_sample_rate.get()),
            "sample_num": int(self.std_sample_num.get()),
        }

    def _ensure_standard_session_lock(self, signature):
        if self.std_session_lock is None:
            self.std_session_lock = dict(signature)
            return

        locked_keys = ("condition_folder", "sample_rate", "sample_num", "pressure_bar", "position_type", "distance_cm", "leak_status", "volume_ml")
        mismatches = [key for key in locked_keys if self.std_session_lock.get(key) != signature.get(key)]
        if mismatches:
            names = ", ".join(mismatches)
            raise ValueError(f"本次 session 已鎖定參數，請勿改變: {names}")

    def _get_single_standard_channel(self):
        channels = []
        for default_ch, var in self.std_channels.items():
            if var.get():
                actual_ch_name = self.std_channel_names[default_ch].get().strip()
                if actual_ch_name:
                    channels.append({'id': actual_ch_name, 'enabled': True})
        if not channels:
            raise ValueError("請至少啟用一個實體通道")

        if self._is_dual_standard_mode():
            labels = {self._tdms_channel_label(ch["id"]) for ch in channels}
            if len(channels) != 2 or labels != {"ai0", "ai1"}:
                raise ValueError("雙通道同步模式請只啟用 ai0 與 ai1")
        elif len(channels) != 1:
            raise ValueError("單通道舊模式每次只能啟用 1 個通道")
        return channels

    def _standard_sync_roles(self, data_dict):
        """Map a true-sync dual-channel capture to the legacy near/far output folders."""
        if len(data_dict) != 2:
            raise ValueError("True-sync Standard capture requires exactly two channels")

        labels = {self._tdms_channel_label(name): name for name in data_dict.keys()}
        if "ai0" in labels and "ai1" in labels:
            return [
                ("near", labels["ai0"], "ai0"),
                ("far", labels["ai1"], "ai1"),
            ]

        ordered = sorted(data_dict.keys(), key=lambda value: self._tdms_channel_label(value))
        return [
            ("near", ordered[0], "ai0"),
            ("far", ordered[1], "ai1"),
        ]

    def _tdms_channel_label(self, channel_name):
        """Use the physical terminal name inside raw single-channel TDMS files."""
        text = str(channel_name or "ai1").strip().replace("\\", "/")
        return text.split("/")[-1] if "/" in text else text

    def _stage_standard_capture(self, role, data_dict, rate, points, capture_timing=None):
        """將單次擷取結果寫入 session staging，等待 near/far 成對輸出。"""
        if role not in ("near", "far"):
            raise ValueError("角色必須是 near 或 far")
        if len(data_dict) != 1:
            raise ValueError("每次只能暫存一個實體通道")

        physical_channel, data = next(iter(data_dict.items()))
        data = np.asarray(data, dtype=np.float32)
        if data.size == 0:
            raise ValueError("擷取資料為空")
        if len(data) != points:
            raise ValueError(f"擷取點數不符: 期望 {points}, 實際 {len(data)}")

        signature = self._standard_signature()
        signature["sample_rate"] = rate
        signature["sample_num"] = points
        self._ensure_standard_session_lock(signature)

        dirs = ConditionFolderHelper.ensure_session_dirs(Path(self.std_folder.get()).expanduser())

        index = len(self.std_staged_records[role]) + 1
        stage_path = dirs["staging"] / self.std_session_id / role / f"{role}_{index:04d}.npz"
        metadata = dict(signature)
        metadata.update({
            "session_id": self.std_session_id,
            "role": role,
            "stage_index": index,
            "physical_channel": physical_channel,
            "capture_timestamp": datetime.now().isoformat(timespec="seconds"),
            "qc_status": "staged",
            "transition_mode": self._transition_enabled(),
        })

        if capture_timing is None:
            end_elapsed = self._standard_timeline_now_elapsed()
            start_elapsed = max(0.0, end_elapsed - (len(data) / float(rate)))
        else:
            start_elapsed, end_elapsed = capture_timing

        metadata.update({
            "capture_start_elapsed_sec": float(start_elapsed),
            "capture_end_elapsed_sec": float(end_elapsed),
            "capture_mid_elapsed_sec": float((float(start_elapsed) + float(end_elapsed)) / 2.0),
        })

        if self._transition_enabled():
            metadata.update(self._transition_segment_info(start_elapsed, end_elapsed))
        else:
            metadata.update({
                "state_label": "",
                "file_label": "",
                "segment_type": "",
                "nearest_event_offset_sec": "",
            })

        metadata.update(self._disturbance_segment_info(start_elapsed, end_elapsed))

        StageIO.save(stage_path, data, metadata)

        record = {"path": str(stage_path), "metadata": metadata}
        self.std_staged_records[role].append(record)
        return record

    def _canonical_dual_data(self, data_dict):
        label_to_key = {}
        for channel_name in data_dict.keys():
            label = self._tdms_channel_label(channel_name)
            if label in ("ai0", "ai1"):
                label_to_key[label] = channel_name

        missing = [label for label in ("ai0", "ai1") if label not in label_to_key]
        if missing:
            raise ValueError(f"雙通道同步缺少通道: {', '.join(missing)}")

        canonical = {
            "ai0": np.asarray(data_dict[label_to_key["ai0"]], dtype=np.float32),
            "ai1": np.asarray(data_dict[label_to_key["ai1"]], dtype=np.float32),
        }
        return canonical, label_to_key

    def _stage_dual_standard_capture(self, data_dict, rate, points, capture_timing=None, extra_metadata=None):
        """將 ai0/ai1 同步擷取結果寫入 dual staging。"""
        canonical, source_channels = self._canonical_dual_data(data_dict)
        for channel, data in canonical.items():
            if data.size == 0:
                raise ValueError(f"{channel}: 擷取資料為空")
            if len(data) != points:
                raise ValueError(f"{channel}: 擷取點數不符: 期望 {points}, 實際 {len(data)}")

        signature = self._standard_signature()
        signature["sample_rate"] = rate
        signature["sample_num"] = points
        self._ensure_standard_session_lock(signature)

        dirs = ConditionFolderHelper.ensure_session_dirs(Path(self.std_folder.get()).expanduser())
        index = len(self.std_staged_records.get("dual", [])) + 1
        stage_path = dirs["staging"] / self.std_session_id / "dual" / f"dual_{index:04d}.npz"
        transition_active = self._transition_enabled()
        metadata = dict(signature)
        metadata.update({
            "session_id": self.std_session_id,
            "role": "dual",
            "stage_index": index,
            "capture_timestamp": datetime.now().isoformat(timespec="seconds"),
            "qc_status": "staged",
            "sync_mode": True,
            "hardware_sync": True,
            "pseudo_sync": False,
            "daq_task": "single_task_multi_channel",
            "sensor_map": "ai0,ai1",
            "sensor_role": "dual",
            "logical_channel": "ai0,ai1",
            "physical_channels": f"{source_channels['ai0']},{source_channels['ai1']}",
            "ai0_physical_channel": source_channels["ai0"],
            "ai1_physical_channel": source_channels["ai1"],
            "ai0_sensor": "ai0",
            "ai1_sensor": "ai1",
            "transition_mode": transition_active,
            "capture_start_elapsed_sec": "",
            "capture_end_elapsed_sec": "",
            "capture_mid_elapsed_sec": "",
            "state_label": "",
            "file_label": "",
            "segment_type": "",
            "nearest_event_offset_sec": "",
        })
        if capture_timing is None:
            end_elapsed = self._standard_timeline_now_elapsed()
            start_elapsed = max(0.0, end_elapsed - (points / float(rate)))
        else:
            start_elapsed, end_elapsed = capture_timing
        
        metadata.update({
            "capture_start_elapsed_sec": float(start_elapsed),
            "capture_end_elapsed_sec": float(end_elapsed),
            "capture_mid_elapsed_sec": float((start_elapsed + end_elapsed) / 2.0),
        })
        
        if transition_active:
            metadata.update(self._transition_segment_info(start_elapsed, end_elapsed))
        else:
            metadata.update({
                "state_label": "",
                "file_label": "",
                "segment_type": "",
                "nearest_event_offset_sec": "",
            })

        metadata.update(self._disturbance_segment_info(start_elapsed, end_elapsed))

        # 連續擷取會帶入：acquisition_mode、first_sample_index、capture_wall_time、毫秒精度的 capture_timestamp
        if extra_metadata:
            metadata.update(extra_metadata)

        StageIO.save_dual(stage_path, canonical, metadata)
        record = {"path": str(stage_path), "metadata": metadata}
        self.std_staged_records.setdefault("dual", []).append(record)
        return record

    # 連續擷取寫進暫存 metadata 的欄位；輸出 TDMS 與 manifest 時一併帶出
    CONTINUITY_META_KEYS = (
        "acquisition_mode",          # "continuous"＝一個 DAQ 任務從頭收到尾；舊版有限取樣沒有這欄
        "first_sample_index",        # 這支檔案第一個點，是任務啟動後第幾個取樣點
        "capture_wall_time",         # 第一個點的電腦時間（Unix 秒，含小數）
        "sample_clock_start_wall",   # 任務啟動（第 0 點）的電腦時間
    )

    def _continuity_meta(self, metadata):
        return {k: metadata.get(k, "") for k in self.CONTINUITY_META_KEYS}

    def _output_stamp_time(self, metadata, fallback_time):
        """
        輸出檔名用的時間。
        連續擷取：用「這支檔案第一個取樣點的真實時間」（毫秒精度）。
        舊資料沒有這個欄位：沿用舊做法（輸出當下時間 + 序號秒數）。
        注意：舊做法的檔名時間是「按下輸出的時間」，不是錄製時間。
        """
        wall = metadata.get("capture_wall_time", "")
        if wall not in ("", None):
            return datetime.fromtimestamp(float(wall))
        return fallback_time

    @staticmethod
    def _stamp_with_ms(dt):
        """YYYYMMDD_HHMMSS_mmm：前段格式不變，後面加毫秒，同一秒的兩支檔案才分得開"""
        return dt.strftime("%Y%m%d_%H%M%S") + f"_{dt.microsecond // 1000:03d}"

    def _build_output_base_name(self, signature, timestamp):
        return (
            f"{signature['leak_status']}_"
            f"{signature['volume_ml']}ml_"
            f"{signature['pressure_bar']}bar_"
            f"{signature['position_type']}{signature['distance_cm']}cm_"
            f"{timestamp}"
        )

    def _unique_output_paths(self, dirs, signature, output_time):
        """保持既有 parser 可讀：若重名就把輸出時間往後推一秒。"""
        candidate = output_time
        for _ in range(3600):
            stamp = candidate.strftime("%Y%m%d_%H%M%S")
            base_name = self._build_output_base_name(signature, stamp)
            paths = {
                "near": dirs["near"] / f"{base_name}.tdms",
                "far": dirs["far"] / f"{base_name}.tdms",
                "merged": dirs["merged_3ch"] / f"{base_name}_with_diff.tdms",
            }
            if not any(path.exists() for path in paths.values()):
                return stamp, base_name, paths
            candidate += timedelta(seconds=1)
        raise RuntimeError("無法產生不重複的輸出檔名")

    def _unique_dual_output_paths(self, dirs, signature, output_time):
        """雙通道同步輸出：ai0、ai1、dual_2ch 三份同名基底。"""
        candidate = output_time
        for _ in range(3600):
            stamp = self._stamp_with_ms(candidate)
            base_name = self._build_output_base_name(signature, stamp)
            paths = {
                "ai0": dirs["ai0"] / f"{base_name}.tdms",
                "ai1": dirs["ai1"] / f"{base_name}.tdms",
                "dual": dirs["dual_2ch"] / f"{base_name}_dual_2ch.tdms",
            }
            if not any(path.exists() for path in paths.values()):
                return stamp, base_name, paths
            candidate += timedelta(milliseconds=1)
        raise RuntimeError("無法產生不重複的 dual 輸出檔名")

    def _write_session_manifest(self, manifest_path, rows):
        fieldnames = [
            "session_id", "dual_index", "condition_folder", "leak_status",
            "volume_ml", "matched_volume_ml", "pressure_bar", "position",
            "sample_rate", "sample_count",
            "capture_start_elapsed_sec", "capture_end_elapsed_sec", "capture_mid_elapsed_sec",
            "ai0_file", "ai1_file",
            "dual_file", "dual_stage", "ai0_physical_channel",
            "ai1_physical_channel", "sync_mode", "pseudo_sync",
            "sensor_map", "qc_status", "output_timestamp",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_dual_session_manifest(self, manifest_path, rows):
        fieldnames = [
            "session_id", "dual_index", "condition_folder", "leak_status",
            "volume_ml", "matched_volume_ml", "pressure_bar", "position",
            "sample_rate", "sample_count",
            "capture_start_elapsed_sec", "capture_end_elapsed_sec", "capture_mid_elapsed_sec",
            "acquisition_mode", "first_sample_index", "capture_wall_time", "sample_clock_start_wall",
            "ai0_file", "ai1_file",
            "dual_file", "dual_stage", "ai0_physical_channel",
            "ai1_physical_channel", "sync_mode", "pseudo_sync",
            "sensor_map", "qc_status", "output_timestamp",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _ensure_transition_dirs(self):
        base = Path(self.std_folder.get()).expanduser() / "transition_single"
        dirs = {
            "base": base,
            "near": base / "near",
            "far": base / "far",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def _ensure_transition_dual_dirs(self):
        base = Path(self.std_folder.get()).expanduser() / "transition_dual"
        dirs = {
            "base": base,
            "ai0": base / "ai0",
            "ai1": base / "ai1",
            "merged": base / "merged_3ch",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

    def _build_transition_output_base_name(self, metadata, timestamp):
        state = metadata.get("state_label", "normal")
        leak_status = "leak1" if state == "leak" else "leak0"
        volume = int(metadata.get("volume_ml", 0)) if state == "leak" else 0
        pressure = int(metadata.get("pressure_bar", 3))
        position_type = metadata.get("position_type", "straight")
        distance = int(metadata.get("distance_cm", 0))
        return f"{leak_status}_{volume}ml_{pressure}bar_{position_type}{distance}cm_{timestamp}"

    def _unique_transition_output_path(self, role_dir, metadata, output_time):
        candidate = output_time
        for _ in range(3600):
            stamp = candidate.strftime("%Y%m%d_%H%M%S")
            base_name = self._build_transition_output_base_name(metadata, stamp)
            path = role_dir / f"{base_name}.tdms"
            if not path.exists():
                return stamp, path
            candidate += timedelta(seconds=1)
        raise RuntimeError("無法產生不重複的 transition 輸出檔名")

    def _unique_transition_dual_output_paths(self, dirs, metadata, output_time):
        candidate = output_time
        for _ in range(3600):
            stamp = self._stamp_with_ms(candidate)
            base_name = self._build_transition_output_base_name(metadata, stamp)
            paths = {
                "ai0": dirs["ai0"] / f"{base_name}.tdms",
                "ai1": dirs["ai1"] / f"{base_name}.tdms",
                "merged": dirs["merged"] / f"{base_name}_with_diff.tdms",
            }
            if not any(path.exists() for path in paths.values()):
                return stamp, base_name, paths
            candidate += timedelta(milliseconds=1)
        raise RuntimeError("無法產生不衝突的 transition dual 輸出檔名")

    def _write_transition_events(self, path, output_time):
        fieldnames = [
            "session_id", "event_index", "event_wall_time", "event_elapsed_sec",
            "from_state", "to_state", "collection_count_at_event", "output_timestamp",
        ]
        rows = []
        for event in getattr(self, "std_transition_events", []):
            row = {key: event.get(key, "") for key in fieldnames}
            row["session_id"] = self.std_session_id
            row["output_timestamp"] = output_time.strftime("%Y%m%d_%H%M%S")
            rows.append(row)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
    def _write_disturbance_events(self, path, output_time):
        fieldnames = [
            "session_id",
            "event_index",
            "disturbance_type",
            "disturbance_strength",
            "start_wall_time",
            "start_elapsed_sec",
            "end_wall_time",
            "end_elapsed_sec",
            "duration_sec",
            "collection_count_at_event",
            "output_timestamp",
        ]
        rows = []
        for event in getattr(self, "std_disturbance_events", []):
            row = {key: event.get(key, "") for key in fieldnames}
            row["session_id"] = self.std_session_id
            row["output_timestamp"] = output_time.strftime("%Y%m%d_%H%M%S")
            rows.append(row)

        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_transition_manifest(self, path, rows):
        fieldnames = [
            "session_id", "output_index", "role", "stage_index", "condition_folder",
            "state_label", "file_label", "segment_type", "capture_start_elapsed_sec",
            "capture_end_elapsed_sec", "capture_mid_elapsed_sec", "nearest_event_offset_sec",
            "disturbance_present", "disturbance_event_count", "disturbance_overlap_sec",
            "disturbance_overlap_ratio", "disturbance_primary_type", "disturbance_primary_strength",
            "leak_status", "volume_ml", "matched_volume_ml", "pressure_bar", "position",
            "sample_rate", "sample_count", "tdms_file", "stage_file", "physical_channel",
            "pseudo_sync", "transition_mode", "output_timestamp",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_transition_dual_manifest(self, path, rows):
        fieldnames = [
            "session_id", "output_index", "stage_index", "condition_folder",
            "state_label", "file_label", "segment_type", "capture_start_elapsed_sec",
            "capture_end_elapsed_sec", "capture_mid_elapsed_sec", "nearest_event_offset_sec",
            "acquisition_mode", "first_sample_index", "capture_wall_time", "sample_clock_start_wall",
            "disturbance_present", "disturbance_event_count", "disturbance_overlap_sec",
            "disturbance_overlap_ratio", "disturbance_primary_type", "disturbance_primary_strength",
            "leak_status", "volume_ml", "matched_volume_ml", "pressure_bar", "position",
            "sample_rate", "sample_count", "ai0_file", "ai1_file", "merged_file",
            "dual_stage", "ai0_physical_channel", "ai1_physical_channel",
            "logical_channel", "sync_mode", "hardware_sync", "pseudo_sync",
            "transition_mode", "qc_status", "output_timestamp",
        ]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _std_output_transition_dual_session(self):
        """Output true-sync dual transition data as ai0, ai1, and merged 3ch."""
        try:
            if self.is_collecting:
                self.std_status.config(text="正在收集中，請先停止再輸出 transition 3ch", foreground="orange")
                return

            dual_count = self._dual_count()
            if dual_count == 0:
                self.std_status.config(text="沒有 transition dual 暫存資料可輸出", foreground="orange")
                return

            transition_dirs = self._ensure_transition_dual_dirs()
            output_time = datetime.now()
            rows = []

            for idx, record in enumerate(self.std_staged_records.get("dual", []), start=1):
                data_dict, metadata = StageIO.load_dual(record["path"])
                if "ai0" not in data_dict or "ai1" not in data_dict:
                    raise ValueError(f"transition dual #{idx}: 缺少 ai0/ai1")

                ai0_data = np.asarray(data_dict["ai0"], dtype=np.float32)
                ai1_data = np.asarray(data_dict["ai1"], dtype=np.float32)
                if len(ai0_data) != len(ai1_data):
                    raise ValueError(f"transition dual #{idx}: ai0/ai1 sample_count 不一致")
                if len(ai0_data) == 0:
                    raise ValueError(f"transition dual #{idx}: 資料長度為 0")

                if not metadata.get("transition_mode"):
                    rate = float(metadata.get("sample_rate", 1) or 1)
                    start_elapsed = float(metadata.get("capture_start_elapsed_sec", 0.0) or 0.0)
                    end_elapsed = float(
                        metadata.get("capture_end_elapsed_sec", len(ai0_data) / rate) or (len(ai0_data) / rate)
                    )
                    metadata["transition_mode"] = True
                    metadata.update(self._transition_segment_info(start_elapsed, end_elapsed))
                    metadata.update(self._disturbance_segment_info(start_elapsed, end_elapsed))

                state_label = metadata.get("state_label") or "normal"
                effective_leak_status = "leak1" if state_label == "leak" else "leak0"
                effective_volume = int(metadata.get("volume_ml", 0)) if state_label == "leak" else 0

                name_meta = dict(metadata)
                name_meta["state_label"] = state_label
                name_meta["volume_ml"] = effective_volume
                stamp, base_name, paths = self._unique_transition_dual_output_paths(
                    transition_dirs,
                    name_meta,
                    self._output_stamp_time(metadata, output_time + timedelta(seconds=idx - 1)),
                )

                ai0_physical = self._tdms_channel_label(metadata.get("ai0_physical_channel", "ai0"))
                ai1_physical = self._tdms_channel_label(metadata.get("ai1_physical_channel", "ai1"))
                diff_data = ai1_data - ai0_data
                position = f"{metadata.get('position_type', 'straight')}{metadata.get('distance_cm', 0)}cm"
                common_meta = dict(metadata)
                common_meta.update({
                    "session_id": self.std_session_id,
                    "output_index": idx,
                    "output_timestamp": stamp,
                    "condition_folder": Path(self.std_folder.get()).name,
                    "leak_status": effective_leak_status,
                    "volume_ml": effective_volume,
                    "matched_volume_ml": metadata.get("matched_volume_ml", metadata.get("volume_ml", 0)),
                    "sample_count": int(len(ai0_data)),
                    "sync_mode": True,
                    "hardware_sync": True,
                    "pseudo_sync": False,
                    "daq_task": "single_task_multi_channel",
                    "sensor_map": "ai0,ai1",
                    "source_dual_stage": record["path"],
                    "transition_mode": True,
                })

                ai0_ok = TDMSSaver.save(
                    paths["ai0"],
                    {"ai0": ai0_data},
                    {
                        **common_meta,
                        "sensor_role": "ai0",
                        "sensor_channel": "ai0",
                        "logical_channel": "ai0",
                        "physical_channel": ai0_physical,
                    },
                )
                ai1_ok = TDMSSaver.save(
                    paths["ai1"],
                    {"ai1": ai1_data},
                    {
                        **common_meta,
                        "sensor_role": "ai1",
                        "sensor_channel": "ai1",
                        "logical_channel": "ai1",
                        "physical_channel": ai1_physical,
                    },
                )
                merged_ok = TDMSSaver.save(
                    paths["merged"],
                    {"ai1": ai1_data, "ai0": ai0_data, "diff": diff_data},
                    {
                        **common_meta,
                        "sensor_role": "merged_3ch",
                        "logical_channel": "ai1,ai0,diff",
                        "ai0_physical_channel": ai0_physical,
                        "ai1_physical_channel": ai1_physical,
                        "source_ai0_file": paths["ai0"].name,
                        "source_ai1_file": paths["ai1"].name,
                    },
                )
                if not (ai0_ok and ai1_ok and merged_ok):
                    raise RuntimeError(f"transition dual #{idx}: TDMS 輸出失敗")

                rows.append({
                    "session_id": self.std_session_id,
                    "output_index": idx,
                    "stage_index": metadata.get("stage_index", idx),
                    "condition_folder": common_meta["condition_folder"],
                    "state_label": state_label,
                    "file_label": metadata.get("file_label", 1 if state_label == "leak" else 0),
                    "segment_type": metadata.get("segment_type", "steady"),
                    "capture_start_elapsed_sec": metadata.get("capture_start_elapsed_sec", ""),
                    "capture_end_elapsed_sec": metadata.get("capture_end_elapsed_sec", ""),
                    "capture_mid_elapsed_sec": metadata.get("capture_mid_elapsed_sec", ""),
                    **self._continuity_meta(metadata),
                    "nearest_event_offset_sec": metadata.get("nearest_event_offset_sec", ""),
                    
                    "disturbance_present": metadata.get("disturbance_present", 0),
                    "disturbance_event_count": metadata.get("disturbance_event_count", 0),
                    "disturbance_overlap_sec": metadata.get("disturbance_overlap_sec", 0.0),
                    "disturbance_overlap_ratio": metadata.get("disturbance_overlap_ratio", 0.0),
                    "disturbance_primary_type": metadata.get("disturbance_primary_type", ""),
                    "disturbance_primary_strength": metadata.get("disturbance_primary_strength", ""),
                    
                    "leak_status": effective_leak_status,
                    "volume_ml": effective_volume,
                    "matched_volume_ml": common_meta["matched_volume_ml"],
                    "pressure_bar": metadata.get("pressure_bar", 3),
                    "position": position,
                    "sample_rate": metadata.get("sample_rate", ""),
                    "sample_count": len(ai0_data),
                    "ai0_file": str(paths["ai0"]),
                    "ai1_file": str(paths["ai1"]),
                    "merged_file": str(paths["merged"]),
                    "dual_stage": record["path"],
                    "ai0_physical_channel": ai0_physical,
                    "ai1_physical_channel": ai1_physical,
                    "logical_channel": "ai1,ai0,diff",
                    "sync_mode": True,
                    "hardware_sync": True,
                    "pseudo_sync": False,
                    "transition_mode": True,
                    "qc_status": "ok",
                    "output_timestamp": stamp,
                })

            stamp = output_time.strftime("%Y%m%d_%H%M%S")
            self._write_transition_events(transition_dirs["base"] / f"transition_dual_events_{stamp}.csv", output_time)
            self._write_transition_dual_manifest(transition_dirs["base"] / f"transition_dual_manifest_{stamp}.csv", rows)

            self._write_disturbance_events(
                transition_dirs["base"] / f"transition_dual_disturbance_events_{stamp}.csv",
                output_time,
            )            

            try:
                shutil.rmtree(Path(self.std_folder.get()).expanduser() / "_session_staging" / self.std_session_id)
            except Exception:
                pass

            self.std_status.config(text=f"已輸出 transition 3ch {len(rows)} 筆", foreground="green")
            self._reset_standard_session()

        except Exception as e:
            ErrorHandler.log_error(e, "輸出 transition dual 3ch")
            self.std_status.config(text=f"transition 3ch 輸出失敗: {str(e)[:80]}", foreground="red")

    def _std_output_transition_session(self):
        """輸出 transition 單通道資料與事件 manifest，不做 near/far pseudo-sync。"""
        if self._is_dual_standard_mode():
            self._std_output_transition_dual_session()
            return
        try:
            if self.is_collecting:
                self.std_status.config(text="收集中不能輸出 transition，請先停止或等待完成", foreground="orange")
                return
            total_count = self._transition_total_count()
            if total_count == 0:
                self.std_status.config(text="沒有 transition 暫存資料可輸出", foreground="orange")
                return

            transition_dirs = self._ensure_transition_dirs()
            output_time = datetime.now()
            manifest_rows = []
            output_index = 0

            records = []
            for role in ("near", "far"):
                for record in self.std_staged_records.get(role, []):
                    records.append((role, record))
            records.sort(key=lambda item: float(item[1]["metadata"].get("capture_mid_elapsed_sec") or item[1]["metadata"].get("stage_index", 0)))

            for role, record in records:
                data, metadata = StageIO.load(record["path"])
                if len(data) == 0:
                    raise ValueError(f"{role} #{metadata.get('stage_index', '?')}: 資料長度為 0")

                if not metadata.get("transition_mode"):
                    rate = float(metadata.get("sample_rate", 1) or 1)
                    start_elapsed = float(metadata.get("capture_start_elapsed_sec", 0.0) or 0.0)
                    end_elapsed = float(metadata.get("capture_end_elapsed_sec", len(data) / rate) or (len(data) / rate))
                    metadata["transition_mode"] = True
                    metadata.update(self._transition_segment_info(start_elapsed, end_elapsed))
                    metadata.update(self._disturbance_segment_info(start_elapsed, end_elapsed))
                    
                output_index += 1
                role_dir = transition_dirs["near"] if role == "near" else transition_dirs["far"]
                stamp, out_path = self._unique_transition_output_path(
                    role_dir,
                    metadata,
                    output_time + timedelta(seconds=output_index - 1),
                )
                physical = self._tdms_channel_label(metadata.get("physical_channel", "ai1"))
                position = f"{metadata.get('position_type', 'straight')}{metadata.get('distance_cm', 0)}cm"
                state_label = metadata.get("state_label", "normal")
                common_meta = dict(metadata)
                common_meta.update({
                    "sensor_role": role,
                    "logical_channel": role,
                    "physical_channel": physical,
                    "pseudo_sync": False,
                    "output_timestamp": stamp,
                    "transition_mode": True,
                })
                ok = TDMSSaver.save(out_path, {physical: data}, common_meta)
                if not ok:
                    raise RuntimeError(f"{role} #{metadata.get('stage_index', '?')}: TDMS 輸出失敗")

                manifest_rows.append({
                    "session_id": self.std_session_id,
                    "output_index": output_index,
                    "role": role,
                    "stage_index": metadata.get("stage_index", ""),
                    "condition_folder": metadata.get("condition_folder", Path(self.std_folder.get()).name),
                    "state_label": state_label,
                    "file_label": metadata.get("file_label", 1 if state_label == "leak" else 0),
                    "segment_type": metadata.get("segment_type", "steady"),
                    "capture_start_elapsed_sec": metadata.get("capture_start_elapsed_sec", ""),
                    "capture_end_elapsed_sec": metadata.get("capture_end_elapsed_sec", ""),
                    "capture_mid_elapsed_sec": metadata.get("capture_mid_elapsed_sec", ""),
                    "nearest_event_offset_sec": metadata.get("nearest_event_offset_sec", ""),
                    
                    "disturbance_present": metadata.get("disturbance_present", 0),
                    "disturbance_event_count": metadata.get("disturbance_event_count", 0),
                    "disturbance_overlap_sec": metadata.get("disturbance_overlap_sec", 0.0),
                    "disturbance_overlap_ratio": metadata.get("disturbance_overlap_ratio", 0.0),
                    "disturbance_primary_type": metadata.get("disturbance_primary_type", ""),
                    "disturbance_primary_strength": metadata.get("disturbance_primary_strength", ""),
                                        
                    "leak_status": "leak1" if state_label == "leak" else "leak0",
                    "volume_ml": int(metadata.get("volume_ml", 0)) if state_label == "leak" else 0,
                    "matched_volume_ml": metadata.get("matched_volume_ml", metadata.get("volume_ml", 0)),
                    "pressure_bar": metadata.get("pressure_bar", 3),
                    "position": position,
                    "sample_rate": metadata.get("sample_rate", ""),
                    "sample_count": len(data),
                    "tdms_file": str(out_path),
                    "stage_file": record["path"],
                    "physical_channel": physical,
                    "pseudo_sync": False,
                    "transition_mode": True,
                    "output_timestamp": stamp,
                })

            stamp = output_time.strftime("%Y%m%d_%H%M%S")
            self._write_transition_events(transition_dirs["base"] / f"transition_events_{stamp}.csv", output_time)
            self._write_transition_manifest(transition_dirs["base"] / f"transition_manifest_{stamp}.csv", manifest_rows)

            self._write_disturbance_events(
                transition_dirs["base"] / f"transition_disturbance_events_{stamp}.csv",
                output_time,
            )
                    
            try:
                shutil.rmtree(Path(self.std_folder.get()).expanduser() / "_session_staging" / self.std_session_id)
            except Exception:
                pass

            self.std_status.config(
                text=f"已輸出 transition 單通道 {len(manifest_rows)} 筆與事件 manifest",
                foreground="green",
            )
            self._reset_standard_session()

        except Exception as e:
            ErrorHandler.log_error(e, "輸出 transition single-channel")
            self.std_status.config(text=f"transition 輸出失敗: {str(e)[:80]}", foreground="red")

    def _std_output_dual_session(self):
        """輸出真同步 ai0/ai1 dual staging，不走 near/far pseudo-sync。"""
        try:
            if self.is_collecting:
                self.std_status.config(text="收集中不能輸出，請先停止或等待完成", foreground="orange")
                return

            dual_count = self._dual_count()
            if dual_count == 0:
                self.std_status.config(text="沒有雙通道同步暫存資料可輸出", foreground="orange")
                return

            dirs = self._prepare_standard_folder()
            if not dirs:
                self.std_status.config(text="資料夾尚未準備完成", foreground="red")
                return

            rows = []
            output_time = datetime.now()
            for idx, record in enumerate(self.std_staged_records.get("dual", []), start=1):
                data_dict, metadata = StageIO.load_dual(record["path"])
                if "ai0" not in data_dict or "ai1" not in data_dict:
                    raise ValueError(f"dual #{idx}: 暫存檔缺少 ai0/ai1")
                ai0_data = np.asarray(data_dict["ai0"], dtype=np.float32)
                ai1_data = np.asarray(data_dict["ai1"], dtype=np.float32)
                if len(ai0_data) != len(ai1_data):
                    raise ValueError(f"dual #{idx}: ai0/ai1 sample_count 不一致")
                if len(ai0_data) == 0:
                    raise ValueError(f"dual #{idx}: 資料長度為 0")

                signature = dict(self.std_session_lock or metadata)
                stamp, base_name, paths = self._unique_dual_output_paths(
                    dirs,
                    signature,
                    self._output_stamp_time(metadata, output_time + timedelta(seconds=idx - 1)),
                )
                common_meta = dict(signature)
                common_meta.update({
                    "session_id": self.std_session_id,
                    "dual_index": idx,
                    "output_timestamp": stamp,
                    "sync_mode": True,
                    "hardware_sync": True,
                    "pseudo_sync": False,
                    "daq_task": "single_task_multi_channel",
                    "sensor_map": "ai0,ai1",
                    "sample_count": int(len(ai0_data)),
                    "condition_folder": Path(self.std_folder.get()).name,
                    "source_dual_stage": record["path"],
                })
                common_meta.update(self._continuity_meta(metadata))

                ai0_physical = metadata.get("ai0_physical_channel", "ai0")
                ai1_physical = metadata.get("ai1_physical_channel", "ai1")
                ai0_ok = TDMSSaver.save(
                    paths["ai0"],
                    {"ai0": ai0_data},
                    {
                        **common_meta,
                        "sensor_role": "ai0",
                        "sensor_channel": "ai0",
                        "physical_channel": ai0_physical,
                        "logical_channel": "ai0",
                    },
                )
                ai1_ok = TDMSSaver.save(
                    paths["ai1"],
                    {"ai1": ai1_data},
                    {
                        **common_meta,
                        "sensor_role": "ai1",
                        "sensor_channel": "ai1",
                        "physical_channel": ai1_physical,
                        "logical_channel": "ai1",
                    },
                )
                dual_ok = TDMSSaver.save(
                    paths["dual"],
                    {"ai0": ai0_data, "ai1": ai1_data},
                    {
                        **common_meta,
                        "sensor_role": "dual_2ch",
                        "logical_channel": "ai0,ai1",
                        "ai0_physical_channel": ai0_physical,
                        "ai1_physical_channel": ai1_physical,
                    },
                )
                if not (ai0_ok and ai1_ok and dual_ok):
                    raise RuntimeError(f"dual #{idx}: TDMS 輸出失敗")

                rows.append({
                    "session_id": self.std_session_id,
                    "dual_index": idx,
                    "condition_folder": common_meta["condition_folder"],
                    "leak_status": common_meta["leak_status"],
                    "volume_ml": common_meta["volume_ml"],
                    "matched_volume_ml": common_meta.get("matched_volume_ml", common_meta["volume_ml"]),
                    "pressure_bar": common_meta["pressure_bar"],
                    "position": f"{common_meta['position_type']}{common_meta['distance_cm']}cm",
                    "sample_rate": common_meta["sample_rate"],
                    "sample_count": len(ai0_data),
                    "capture_start_elapsed_sec": metadata.get("capture_start_elapsed_sec", ""),
                    "capture_end_elapsed_sec": metadata.get("capture_end_elapsed_sec", ""),
                    "capture_mid_elapsed_sec": metadata.get("capture_mid_elapsed_sec", ""),
                    **self._continuity_meta(metadata),
                    "ai0_file": str(paths["ai0"]),
                    "ai1_file": str(paths["ai1"]),
                    "dual_file": str(paths["dual"]),
                    "dual_stage": record["path"],
                    "ai0_physical_channel": ai0_physical,
                    "ai1_physical_channel": ai1_physical,
                    "sync_mode": True,
                    "pseudo_sync": False,
                    "sensor_map": common_meta["sensor_map"],
                    "qc_status": "ok",
                    "output_timestamp": stamp,
                })

            manifest_path = Path(self.std_folder.get()) / f"dual_session_manifest_{output_time.strftime('%Y%m%d_%H%M%S')}.csv"
            self._write_dual_session_manifest(manifest_path, rows)
            
            stamp = output_time.strftime("%Y%m%d_%H%M%S")
            disturbance_path = Path(self.std_folder.get()) / f"disturbance_events_{stamp}.csv"
            self._write_disturbance_events(disturbance_path, output_time)

            try:
                shutil.rmtree(Path(self.std_folder.get()) / "_session_staging" / self.std_session_id)
            except Exception:
                pass

            self.std_status.config(text=f"已輸出 {len(rows)} 筆 dual_2ch", foreground="green")
            self._reset_standard_session()

        except Exception as e:
            ErrorHandler.log_error(e, "輸出 dual_2ch")
            self.std_status.config(text=f"dual 輸出失敗: {str(e)[:80]}", foreground="red")

    def _std_output_session(self):
        """將 near/far 暫存結果一次輸出到 near、far、merged_3ch。"""
        try:
            if self._dual_count() > 0 or self._is_dual_standard_mode():
                if self._transition_enabled():
                    self.std_status.config(text="transition 模式請使用「輸出 transition 3ch」", foreground="orange")
                    return
                self._std_output_dual_session()
                return
            if self._transition_enabled():
                self.std_status.config(text="transition 模式請使用「輸出 transition single-channel」", foreground="orange")
                return
            near_count, far_count = self._standard_counts()
            if near_count == 0 or far_count == 0:
                self.std_status.config(text="near/far 都至少需要 1 筆", foreground="orange")
                return
            pair_count = min(near_count, far_count)

            dirs = self._prepare_standard_folder()
            if not dirs:
                self.std_status.config(text="資料夾尚未準備完成", foreground="red")
                return

            rows = []
            output_time = datetime.now()
            used_near_records = self.std_staged_records["near"][:pair_count]
            used_far_records = self.std_staged_records["far"][:pair_count]

            for idx, (near_rec, far_rec) in enumerate(zip(used_near_records, used_far_records), start=1):
                near_data, near_meta = StageIO.load(near_rec["path"])
                far_data, far_meta = StageIO.load(far_rec["path"])

                if int(near_meta["sample_rate"]) != int(far_meta["sample_rate"]):
                    raise ValueError(f"pair {idx}: near/far sample_rate 不一致")
                if len(near_data) != len(far_data):
                    raise ValueError(f"pair {idx}: near/far sample_count 不一致")
                if len(near_data) == 0:
                    raise ValueError(f"pair {idx}: 資料長度為 0")

                signature = dict(self.std_session_lock or near_meta)
                stamp, base_name, paths = self._unique_output_paths(dirs, signature, output_time + timedelta(seconds=idx - 1))
                diff_data = near_data - far_data

                common_meta = dict(signature)
                common_meta.update({
                    "session_id": self.std_session_id,
                    "pair_index": idx,
                    "output_timestamp": stamp,
                    "pseudo_sync": not (near_meta.get("sync_mode") and far_meta.get("sync_mode")),
                    "sync_mode": bool(near_meta.get("sync_mode") and far_meta.get("sync_mode")),
                    "daq_task": near_meta.get("daq_task", ""),
                    "sensor_map": near_meta.get("sensor_map", ""),
                    "sample_count": int(len(near_data)),
                    "condition_folder": Path(self.std_folder.get()).name,
                })

                near_physical = self._tdms_channel_label(near_meta.get("physical_channel", "ai1"))
                far_physical = self._tdms_channel_label(far_meta.get("physical_channel", "ai1"))
                near_ok = TDMSSaver.save(paths["near"], {near_physical: near_data}, {**common_meta, "sensor_role": "near", "physical_channel": near_physical, "logical_channel": "near", "pseudo_sync": False})
                far_ok = TDMSSaver.save(paths["far"], {far_physical: far_data}, {**common_meta, "sensor_role": "far", "physical_channel": far_physical, "logical_channel": "far", "pseudo_sync": False})
                merged_ok = TDMSSaver.save(paths["merged"], {"ai1": near_data, "ai0": far_data, "diff": diff_data}, {**common_meta, "sensor_role": "merged_3ch", "logical_channel": "ai1,ai0,diff", "source_near_file": paths["near"].name, "source_far_file": paths["far"].name})

                if not (near_ok and far_ok and merged_ok):
                    raise RuntimeError(f"pair {idx}: TDMS 輸出失敗")

                rows.append({
                    "session_id": self.std_session_id,
                    "pair_index": idx,
                    "condition_folder": common_meta["condition_folder"],
                    "leak_status": common_meta["leak_status"],
                    "volume_ml": common_meta["volume_ml"],
                    "matched_volume_ml": common_meta.get("matched_volume_ml", common_meta["volume_ml"]),
                    "pressure_bar": common_meta["pressure_bar"],
                    "position": f"{common_meta['position_type']}{common_meta['distance_cm']}cm",
                    "sample_rate": common_meta["sample_rate"],
                    "sample_count": len(near_data),
                    "near_file": str(paths["near"]),
                    "far_file": str(paths["far"]),
                    "merged_file": str(paths["merged"]),
                    "near_stage": near_rec["path"],
                    "far_stage": far_rec["path"],
                    "near_physical_channel": near_physical,
                    "far_physical_channel": far_physical,
                    "pseudo_sync": common_meta["pseudo_sync"],
                    "qc_status": "ok",
                    "output_timestamp": stamp,
                })

            manifest_path = Path(self.std_folder.get()) / f"session_manifest_{output_time.strftime('%Y%m%d_%H%M%S')}.csv"
            self._write_session_manifest(manifest_path, rows)

            discarded_near = max(0, near_count - pair_count)
            discarded_far = max(0, far_count - pair_count)

            # 正式輸出成功後清掉整個本次 staging；多收的未配對檔不保留，避免下次配對混亂。
            try:
                shutil.rmtree(Path(self.std_folder.get()) / "_session_staging" / self.std_session_id)
            except Exception:
                pass

            if discarded_near or discarded_far:
                self.std_status.config(
                    text=f"已輸出 {len(rows)} 組；未配對 near {discarded_near} 筆、far {discarded_far} 筆已刪除",
                    foreground="orange",
                )
            else:
                self.std_status.config(text=f"已輸出 {len(rows)} 組 near/far/merged", foreground="green")
            self._reset_standard_session()

        except Exception as e:
            ErrorHandler.log_error(e, "輸出 near/far/merged")
            self.std_status.config(text=f"輸出失敗: {str(e)[:80]}", foreground="red")
    
    def _update_std_device_mode(self):
        """更新設備模式狀態"""
        if self.std_simulation_mode.get():
            self.std_device_status.config(
                text="[模擬] 測試模式", 
                foreground="orange")
            self.std_device_name.config(state='disabled')
        else:
            device_name = self.std_device_name.get().strip()
            if NIDAQ_AVAILABLE:
                self.std_device_status.config(
                    text=f"[實際] {device_name}", 
                    foreground="green")
            else:
                self.std_device_status.config(
                    text=f"NI-DAQ 未載入: {(NIDAQ_IMPORT_ERROR or 'unknown')[:90]}",
                    foreground="red")
            self.std_device_name.config(state='normal')
    
    def _update_std_filename(self):
        """更新檔名預覽"""
        try:
            leak_status = self.std_leak_status.get()
            volume = "0" if leak_status == "leak0" else self.std_volume.get()
            pressure = self.std_pressure.get()
            pos_type = self.std_position_type.get()
            pos_dist = self.std_position_dist.get()
            
            filename = f"{leak_status}_{volume}ml_{pressure}bar_{pos_type}{pos_dist}cm.tdms"
            self.std_filename_preview.config(text=filename)
        except:
            self.std_filename_preview.config(text="(檢查輸入)")
    
    def _update_std_params(self, changed):
        """更新採樣參數"""
        try:
            rate = float(self.std_sample_rate.get())
            
            if changed == 'time':
                duration = float(self.std_duration.get())
                points = int(rate * duration)
                self.std_sample_num.delete(0, tk.END)
                self.std_sample_num.insert(0, str(points))
            
            elif changed == 'points':
                points = int(self.std_sample_num.get())
                duration = points / rate
                self.std_duration.delete(0, tk.END)
                self.std_duration.insert(0, f"{duration:.4f}")
        except:
            pass

    def _estimate_completion_interval_count(self, total_seconds, interval_seconds, sample_rate, sample_num):
        """Estimate count when total duration means wait-time budget only."""
        capture_seconds = sample_num / sample_rate if sample_rate > 0 else 0
        effective_cycle_sec = (sample_num / sample_rate) + interval_seconds
        count = int(total_seconds // effective_cycle_sec) if effective_cycle_sec > 0 else 0
        estimated_wall_seconds = total_seconds + count * capture_seconds
        return count, capture_seconds, estimated_wall_seconds
    
    def _update_std_estimate(self):
        """更新預估"""
        try:
            duration = float(self.std_long_duration.get())
            if self.std_long_unit.get() == "分":
                duration *= 60
            elif self.std_long_unit.get() == "時":
                duration *= 3600
            
            interval = float(self.std_long_interval.get())
            if self.std_long_interval_unit.get() == "分":
                interval *= 60
            
            if interval < 0:
                self.std_estimate.config(text="預估: 間隔必須 >= 0")
                return

            rate = float(self.std_sample_rate.get())
            points = int(self.std_sample_num.get())
            count, capture_sec, estimated_wall_sec = self._estimate_completion_interval_count(
                duration, interval, rate, points
            )
            self.std_estimate.config(
                text=f"預估: {count} 筆 (完成後等{interval:.1f}秒；實際總耗時約{estimated_wall_sec:.0f}秒+暫存)"
            )
        except:
            self.std_estimate.config(text="預估: -- 筆")
    
    def _std_single_capture(self):
        """單次擷取"""
        self.std_status.config(text="擷取中...", foreground="blue")
        self.btn_std_single_capture.config(state='disabled')
        
        if self.std_session_start_time is None:
            now = time.time()
            self.std_session_start_time = now
            self.std_timeline_start_time = now
        def capture():
            try:
                # 參數驗證
                rate = int(self.std_sample_rate.get())
                points = int(self.std_sample_num.get())
                duration = points / rate
                
                valid, msg = ErrorHandler.validate_params(rate, points, duration)
                if not valid:
                    self.root.after(0, lambda: self.std_status.config(
                        text=f"參數錯誤: {msg.split(':')[1].split('•')[1].strip() if '•' in msg else '請檢查參數'}", 
                        foreground="red"))
                    return
                
                # 配置設備
                device_name = self.std_device_name.get().strip()
                simulation_mode = self.std_simulation_mode.get()
                self.multi_capture.configure_device(device_name, simulation_mode)
                
                # 雙通道同步模式會在同一個 NI-DAQ task 中讀 ai0/ai1；舊模式才使用 near/far 角色。
                channels = self._get_single_standard_channel()
                self.multi_capture.configure_channels(channels)
                dual_mode = self._is_dual_standard_mode()
                role = "dual" if dual_mode else self.std_sensor_role.get()
                
                # 更新狀態
                mode_text = "模擬模式" if simulation_mode else f"設備: {device_name}"
                self.root.after(0, lambda: self.std_device_status.config(
                    text=f"[擷取] {mode_text}", foreground="blue"))
                
                # 擷取（帶重試）
                def do_capture():
                    return self.multi_capture.capture_sync(rate, points)
                
                capture_start_elapsed = self._standard_timeline_now_elapsed()
                self.current_data = SmartRetry.retry_with_backoff(do_capture)
                capture_end_elapsed = self._standard_timeline_now_elapsed()
                capture_timing = (capture_start_elapsed, capture_end_elapsed)
                if dual_mode:
                    staged = self._stage_dual_standard_capture(
                        self.current_data,
                        rate,
                        points,
                        capture_timing=capture_timing,
                    )
                    staged_text = "dual #{}".format(staged["metadata"]["stage_index"])
                else:
                    staged = self._stage_standard_capture(role, self.current_data, rate, points, capture_timing=capture_timing)
                    staged_text = "{} #{}".format(role, staged["metadata"]["stage_index"])
                
                # 更新顯示
                self.root.after(0, lambda: self.waveform_viewer.plot_waveforms(
                    self.current_data, rate))
                self.root.after(0, lambda text=staged_text: self.std_status.config(
                    text=f"✓ 已暫存 {text}", 
                    foreground="green"))
                self.root.after(0, self._update_standard_session_status)
                self.root.after(0, self._update_transition_status)
                
            except Exception as e:
                ErrorHandler.log_error(e, "單次擷取")
                error_msg = str(e)
                # 簡化錯誤訊息
                if "擷取失敗" in error_msg:
                    display_msg = error_msg.split('\n')[0]  # 只顯示第一行
                else:
                    display_msg = f"擷取失敗: {error_msg[:50]}"
                
                self.root.after(0, lambda: self.std_status.config(
                    text=display_msg, foreground="red"))
            finally:
                self.root.after(0, lambda: self.btn_std_single_capture.config(state='normal'))
        
        threading.Thread(target=capture, daemon=True).start()
    
    def _std_single_save(self):
        """儲存單次數據"""
        if not hasattr(self, 'current_data'):
            self.std_status.config(text="沒有數據可儲存", foreground="orange")
            return
        
        try:
            folder = Path(self.std_folder.get())
            leak_status = self.std_leak_status.get()
            subfolder = folder / leak_status
            subfolder.mkdir(parents=True, exist_ok=True)
            
            filename = self.std_filename_preview.cget("text")
            base_name = filename.replace('.tdms', '')
            counter = 1
            final_path = subfolder / filename
            
            while final_path.exists():
                filename = f"{base_name}_{counter:03d}.tdms"
                final_path = subfolder / filename
                counter += 1
            
            metadata = {
                'leak_status': leak_status,
                'leak_volume': self.std_volume.get() if leak_status == 'leak1' else '0',
                'pressure': self.std_pressure.get() + 'bar',
                'leak_position': f"{self.std_position_type.get()}{self.std_position_dist.get()}cm",
                'sample_rate': int(self.std_sample_rate.get()),
                'sample_num': int(self.std_sample_num.get()),
                'n_channels': len(self.current_data),
                'timestamp': datetime.now().isoformat()
            }
            
            success = TDMSSaver.save(final_path, self.current_data, metadata)
            
            if success:
                self.std_status.config(
                    text=f"已儲存: {final_path.name}", 
                    foreground="green")
                self.btn_std_single_save.config(state='disabled')
            
        except Exception as e:
            ErrorHandler.log_error(e, "儲存單次數據")
            self.std_status.config(
                text=f"儲存失敗: {str(e)[:30]}", 
                foreground="red")
    
    def _std_long_start(self):
        """開始長時間收集"""
        try:
            # 參數
            duration = float(self.std_long_duration.get())
            if self.std_long_unit.get() == "分":
                duration *= 60
            elif self.std_long_unit.get() == "時":
                duration *= 3600
            
            interval = float(self.std_long_interval.get())
            if self.std_long_interval_unit.get() == "分":
                interval *= 60
            if duration <= 0 or interval < 0:
                self.std_status.config(text="總時長必須 > 0，間隔必須 >= 0", foreground="red")
                return
            
            rate = int(self.std_sample_rate.get())
            points = int(self.std_sample_num.get())
            n_files, capture_sec, estimated_wall_sec = self._estimate_completion_interval_count(
                duration, interval, rate, points
            )
            
            
            # 雙通道 + 間隔 0 → 連續擷取：一個 DAQ 任務從頭收到尾，檔案之間不漏資料
            # 間隔 > 0 → 沿用舊的「收一筆、等一下、再收一筆」（本來就是要間歇取樣）
            use_continuous = self._is_dual_standard_mode() and interval == 0
            if use_continuous:
                n_files = int(duration // (points / rate))
            
            # 參數驗證
            valid, msg = ErrorHandler.validate_params(rate, points, points/rate)
            if not valid:
                self.std_status.config(text="參數錯誤", foreground="red")
                return
            
            # 磁碟空間檢查
            folder = self.std_folder.get()
            if n_files == 0:
                self.std_status.config(text="間隔太大，至少需要1筆", foreground="red")
                return
            
            enabled_channels = sum(1 for var in self.std_channels.values() if var.get())
            required_mb = n_files * points * 4 * max(1, enabled_channels) * 1.5 / 1024 / 1024
            
            ok, msg = ErrorHandler.check_disk_space(folder, required_mb)
            if not ok:
                self.std_status.config(text="磁碟空間不足", foreground="red")
                return
            
            # 配置設備
            device_name = self.std_device_name.get().strip()
            simulation_mode = self.std_simulation_mode.get()
            self.multi_capture.configure_device(device_name, simulation_mode)
            
            # 雙通道同步模式會在同一個 NI-DAQ task 中讀 ai0/ai1；舊模式才使用 near/far 角色。
            channels = self._get_single_standard_channel()
            self.multi_capture.configure_channels(channels)
            dual_mode = self._is_dual_standard_mode()
            role = "dual" if dual_mode else self.std_sensor_role.get()
            show_last = bool(self.std_show_last.get())
            
            # 準備
            self._prepare_standard_folder()
            
            # 更新UI
            self.btn_std_long_start.config(state='disabled')
            self.btn_std_long_stop.config(state='normal')
            self.is_collecting = True
            self._update_standard_session_status()
            self.collection_count = 0
            if self.std_session_start_time is None:
                now = time.time()
                self.std_session_start_time = now
                self.std_timeline_start_time = now
            
            if self._transition_enabled():
                self.std_transition_initial_state = self.std_transition_state.get()
                self._update_transition_status()

            # 啟動進度追蹤
            self.progress_tracker.reset()
            self.progress_tracker.start(n_files)
            
            # 顯示開始資訊
            self.std_status.config(
                text=f"開始長時間收集 (等待總時長: {duration}秒, 完成後等待: {interval}秒, 擷取另計, 預估: {n_files}筆)",
                foreground="blue")
            
            # 開始收集
            def collect_loop():
                if use_continuous:
                    # 連續擷取；出錯時錯誤訊息要在「完成」之後顯示，才不會被「完成! 共 0 筆」蓋掉
                    try:
                        summary = self._std_continuous_collect(rate, points, n_files, show_last)
                        err = summary["error"]
                        n_written = summary["n_written"]
                    except Exception as exc:          # 連續擷取還沒開始就失敗（例如找不到讀取器）
                        traceback.print_exc()
                        ErrorHandler.log_error(exc, "連續擷取（啟動前）")
                        err, n_written = exc, 0
                    self.root.after(0, self._std_long_complete)
                    if err is not None:
                        self.root.after(
                            0,
                            lambda e=err, n=n_written: self.std_status.config(
                                text=f"連續擷取失敗（已存 {n} 支）：{type(e).__name__}: {str(e)[:80]}",
                                foreground="red",
                            )
                        )
                    return
            
                start_time = time.time()
                
                for run_index in range(n_files):
                    if not self.is_collecting:
                        break

                    try:
                        self.root.after(0, lambda: self.progress_tracker.set_current_status(
                            "正在擷取..."))
                        
                        # 擷取（帶重試）
                        def do_capture():
                            return self.multi_capture.capture_sync(rate, points)
                        
                        def on_retry(attempt, max_attempts, wait_time, error):
                            self.root.after(0, lambda: self.progress_tracker.set_current_status(
                                f"重試 {attempt}/{max_attempts}... (等待{wait_time:.1f}秒)"))
                        
                        capture_start_elapsed = self._standard_timeline_now_elapsed()
                        data = SmartRetry.retry_with_backoff(
                            do_capture, 
                            max_retries=3,
                            on_retry=on_retry
                        )
                        capture_end_elapsed = self._standard_timeline_now_elapsed()
                        # 更新波形顯示（如果勾選）
                        if show_last:
                            self.current_data = data
                            self.root.after(0, lambda: self.waveform_viewer.plot_waveforms(
                                data, rate))
                        
                        # 暫存，等待 near/far 成對輸出。
                        self.root.after(0, lambda: self.progress_tracker.set_current_status(
                            "正在暫存..."))
                        capture_timing = (capture_start_elapsed, capture_end_elapsed)
                        if dual_mode:
                            staged = self._stage_dual_standard_capture(
                                data,
                                rate,
                                points,
                                capture_timing=capture_timing,
                            )
                        else:
                            staged = self._stage_standard_capture(role, data, rate, points, capture_timing=capture_timing)
                        
                        self.collection_count += 1
                        
                        # 計算檔案大小
                        stage_path = Path(staged["path"])
                        file_size_mb = stage_path.stat().st_size / 1024 / 1024 if stage_path.exists() else 0
                        
                        # 更新進度
                        self.root.after(
                            0,
                            lambda count=self.collection_count, name=stage_path.name, size=file_size_mb:
                                self.progress_tracker.update(count, name, True, size)
                        )
                        self.root.after(0, self._update_standard_session_status)
                        self.root.after(0, self._update_transition_status)
                        
                        # 更新狀態
                        elapsed = time.time() - start_time
                        remaining_wait = max(0.0, (n_files - self.collection_count) * interval)
                        self.root.after(
                            0,
                            lambda count=self.collection_count, used=int(elapsed), rem=int(remaining_wait):
                                self.std_status.config(
                                    text=f"已收集 {count}/{n_files} 筆 | 已用: {used}秒 | 剩餘等待: {rem}秒",
                                    foreground="blue"
                                )
                        )
                    
                    except Exception as e:
                        ErrorHandler.log_error(e, f"長時間收集 #{self.collection_count}")
                        self.collection_count += 1
                        self.root.after(
                            0,
                            lambda count=self.collection_count:
                                self.progress_tracker.update(count, "error.tdms", False, 0)
                        )

                    if self.is_collecting and run_index < n_files - 1:
                        if interval > 0:
                            wait_until = time.time() + interval
                            self.root.after(0, lambda: self.progress_tracker.set_current_status(
                                f"完成，等待 {interval:.1f} 秒..."))
                            while time.time() < wait_until and self.is_collecting:
                                time.sleep(min(0.2, wait_until - time.time()))
                        else:
                            self.root.after(0, lambda: self.progress_tracker.set_current_status(
                                "完成，立即進入下一次擷取..."))
                
                # 完成
                self.root.after(0, self._std_long_complete)
            
            threading.Thread(target=collect_loop, daemon=True).start()
            
        except Exception as e:
            ErrorHandler.log_error(e, "啟動長時間收集")
            self.std_status.config(text=f"啟動失敗: {str(e)[:30]}", foreground="red")
    
    def _std_continuous_collect(self, rate, points, n_files, show_last):
        """
        連續擷取 n_files 支、每支 points 點（雙通道）。在背景執行緒中呼叫。
    
        和舊版的差別：
          - 舊版每支檔案都「開一個新 DAQ 任務 → 收 0.5 秒 → 關掉 → 存檔」，
            開關任務與存檔期間沒有在收，檔案之間會漏 0.6～0.7 秒。
          - 新版只開一次任務、連續取樣，讀取和存檔分開兩個執行緒，檔案首尾相接。
          - 每支檔案的開始時間 = 任務啟動時間 + 取樣序號 / 取樣率，
            不再用「呼叫擷取前的電腦時間」，所以不會受存檔、畫面更新影響。
    
        量程、耦合、端子設定和舊版有限取樣相同（±10 V、DAQmx 預設），
        確保新舊資料的訊號條件一致。
        """
        if AEsignal_DualChannelContinuousReader is None or ContinuousChunkRecorder is None:
            raise RuntimeError(f"nidaq_capture 連續讀取器無法使用：{NIDAQ_IMPORT_ERROR}")
    
        device_name = self.std_device_name.get().strip()
        simulation_mode = bool(self.std_simulation_mode.get())
    
        # 讀取器把第 1 個通道回傳成 "ai0"、第 2 個回傳成 "ai1"，所以先依 ai0、ai1 排好
        channel_ids = sorted(
            (ch["id"] for ch in self._get_single_standard_channel()),
            key=self._tdms_channel_label,
        )
        full_names = [cid if "/" in cid else f"{device_name}/{cid}" for cid in channel_ids]
    
        reader = AEsignal_DualChannelContinuousReader(
            samplerate=rate,
            channelnames=full_names,
            chunk_size=points,             # 每次讀一整支檔案的點數（0.5 秒）
            simulation=simulation_mode,
            daq_buffer_seconds=10.0,       # DAQ 緩衝 10 秒：存檔偶爾卡一下也不會溢位
            min_val=-10.0,
            max_val=10.0,
            coupling=None,
            terminal_config=None,
        )
        chunk_sec = points / float(rate)

        def on_chunk(chunk_number, first_sample_index, chunk):
            # 這支檔案第一個點的真實時間，以及在 session 時間軸上的位置
            start_wall = reader.start_wall_time + first_sample_index / float(rate)
            start_elapsed = start_wall - self.std_session_start_time
            end_elapsed = start_elapsed + chunk_sec
    
            data = {full_names[0]: chunk["ai0"], full_names[1]: chunk["ai1"]}
            extra = {
                "acquisition_mode": "continuous",
                "first_sample_index": int(first_sample_index),
                "capture_wall_time": float(start_wall),
                "sample_clock_start_wall": float(reader.start_wall_time),
                "capture_timestamp": datetime.fromtimestamp(start_wall).isoformat(timespec="milliseconds"),
            }
            staged = self._stage_dual_standard_capture(
                data, rate, points,
                capture_timing=(start_elapsed, end_elapsed),
                extra_metadata=extra,
            )
            self.collection_count += 1
    
            # 每 2 秒才更新一次波形，避免畫圖拖慢存檔
            if show_last and chunk_number % max(1, int(round(2.0 / chunk_sec))) == 0:
                self.current_data = data
                self.root.after(0, lambda d=data: self.waveform_viewer.plot_waveforms(d, rate))
    
            stage_path = Path(staged["path"])
            size_mb = stage_path.stat().st_size / 1024 / 1024 if stage_path.exists() else 0
            self.root.after(
                0,
                lambda count=self.collection_count, name=stage_path.name, size=size_mb:
                    self.progress_tracker.update(count, name, True, size)
            )
            self.root.after(0, self._update_standard_session_status)
            self.root.after(0, self._update_transition_status)
            self.root.after(
                0,
                lambda count=self.collection_count: self.std_status.config(
                    text=f"連續擷取中 {count}/{n_files} 支（{count * chunk_sec:.1f} 秒，無空檔）",
                    foreground="blue",
                )
            )
    
        self.root.after(0, lambda: self.progress_tracker.set_current_status("連續擷取中（單一 DAQ 任務）..."))
        summary = ContinuousChunkRecorder(
            reader,
            n_chunks=n_files,
            on_chunk=on_chunk,
            should_continue=lambda: self.is_collecting,
        ).run()
    
        # 出錯時把完整錯誤印到 Spyder 主控台並寫進 error.log；畫面上的紅字由 collect_loop 負責顯示
        if summary["error"] is not None:
            err = summary["error"]
            print(f"\n[連續擷取錯誤] 已存 {summary['n_written']} 支後停止")
            traceback.print_exception(type(err), err, err.__traceback__)
            ErrorHandler.log_error(err, "連續擷取")
        return summary
    
    def _std_long_stop(self):
        """停止長時間收集"""
        self.is_collecting = False
        self.btn_std_long_start.config(state='normal')
        self.btn_std_long_stop.config(state='disabled')
        self.progress_tracker.stop()
        self._update_standard_session_status()
        self.std_status.config(text="已停止", foreground="orange")
    
    def _std_long_complete(self):
        """長時間收集完成"""
        self.btn_std_long_start.config(state='normal')
        self.btn_std_long_stop.config(state='disabled')
        self.progress_tracker.complete()
        self.is_collecting = False
        self._update_transition_status()
        self._update_standard_session_status()
        self.std_status.config(
            text=f"完成! 共 {self.collection_count} 筆", 
            foreground="green")
    
    # ==================== GAN噪音收集頁面（簡化版）====================
    
    def _build_gan_page(self, parent):
        """建構GAN噪音收集頁面"""
        
        # 使用grid布局
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)
        
        # 左側使用Canvas+Scrollbar實現滾動
        left_container = ttk.Frame(parent)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # 創建Canvas和Scrollbar
        canvas_gan = tk.Canvas(left_container, highlightthickness=0)
        scrollbar_gan = ttk.Scrollbar(left_container, orient="vertical", command=canvas_gan.yview)
        left_panel = ttk.Frame(canvas_gan)
        
        # 配置滾動
        left_panel.bind(
            "<Configure>",
            lambda e: canvas_gan.configure(scrollregion=canvas_gan.bbox("all"))
        )
        
        canvas_gan.create_window((0, 0), window=left_panel, anchor="nw")
        canvas_gan.configure(yscrollcommand=scrollbar_gan.set)
        
        # 佈局
        canvas_gan.pack(side="left", fill="both", expand=True)
        scrollbar_gan.pack(side="right", fill="y")
        
        # 右側面板
        right_panel = ttk.Frame(parent)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # ===== 左側:設定區 =====
        
        # 設備配置（壓縮版）
        lf_device_gan = ttk.LabelFrame(left_panel, text="設備", padding=3)
        lf_device_gan.pack(fill="x", pady=1)
        
        # 設備名稱和模擬模式在同一行
        dev_row_gan = ttk.Frame(lf_device_gan)
        dev_row_gan.pack(fill="x", pady=1)
        ttk.Label(dev_row_gan, text="設備:", width=5).pack(side="left")
        self.gan_device_name = ttk.Entry(dev_row_gan, width=12)
        self.gan_device_name.insert(0, "cDAQ1Mod1")
        self.gan_device_name.pack(side="left", padx=2)
        
        self.gan_simulation_mode = tk.BooleanVar(value=not NIDAQ_AVAILABLE)
        ttk.Checkbutton(dev_row_gan, text="模擬", 
                       variable=self.gan_simulation_mode,
                       command=self._update_gan_device_mode).pack(side="left", padx=2)
        
        self.gan_device_status = ttk.Label(lf_device_gan, text="", 
                                           font=('Arial', 7), foreground="gray")
        self.gan_device_status.pack(pady=1)
        
        # 噪音類型
        lf_noise = ttk.LabelFrame(left_panel, text="噪音類型", padding=10)
        lf_noise.pack(fill="x", pady=5)
        
        # 儲存資料夾
        folder_frame = ttk.Frame(lf_noise)
        folder_frame.pack(fill="x", pady=2)
        ttk.Label(folder_frame, text="儲存資料夾:").pack(side="left")
        self.gan_folder = ttk.Entry(folder_frame, width=20)
        self.gan_folder.insert(0, "./gan_noise_dataset")
        self.gan_folder.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(folder_frame, text="瀏覽", width=6,
                  command=lambda: self._browse_folder(self.gan_folder)).pack(side="right")
        
        # 噪音標籤
        ttk.Label(lf_noise, text="噪音標籤:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,5))
        self.gan_noise_type = tk.StringVar(value="background_clean")
        ttk.Radiobutton(lf_noise, text="background_clean (無洩漏背景音)", 
                       variable=self.gan_noise_type, value="background_clean",
                       command=self._update_gan_filename).pack(anchor="w", padx=10)
        ttk.Radiobutton(lf_noise, text="background_leak (含洩漏背景音)", 
                       variable=self.gan_noise_type, value="background_leak",
                       command=self._update_gan_filename).pack(anchor="w", padx=10)
        
        # 壓力
        pressure_frame = ttk.Frame(lf_noise)
        pressure_frame.pack(fill="x", pady=5)
        ttk.Label(pressure_frame, text="壓力:").pack(side="left")
        self.gan_pressure = ttk.Entry(pressure_frame, width=10)
        self.gan_pressure.insert(0, "3")
        self.gan_pressure.pack(side="left", padx=5)
        ttk.Label(pressure_frame, text="bar").pack(side="left")
        self.gan_pressure.bind('<KeyRelease>', lambda e: self._update_gan_filename())
        
        # 檔名預覽
        ttk.Separator(lf_noise, orient='horizontal').pack(fill="x", pady=5)
        ttk.Label(lf_noise, text="預覽檔名:", font=('Arial', 9, 'bold')).pack(anchor="w")
        self.gan_filename_preview = ttk.Label(lf_noise, text="", foreground="blue",
                                             font=('Courier', 9))
        self.gan_filename_preview.pack(anchor="w", pady=2)
        
        # 擷取參數
        lf_params = ttk.LabelFrame(left_panel, text="⚙️ 擷取參數", padding=10)
        lf_params.pack(fill="x", pady=5)
        
        # 採樣率
        rate_frame = ttk.Frame(lf_params)
        rate_frame.pack(fill="x", pady=2)
        ttk.Label(rate_frame, text="採樣率(Hz):").pack(side="left")
        self.gan_sample_rate = ttk.Entry(rate_frame, width=12)
        self.gan_sample_rate.insert(0, "400000")
        self.gan_sample_rate.pack(side="left", padx=5)
        ttk.Label(rate_frame, text="🔒", foreground="gray").pack(side="left")
        self.gan_sample_rate.bind('<KeyRelease>', lambda e: self._update_gan_params('rate'))
        
        # 擷取時間
        time_frame = ttk.Frame(lf_params)
        time_frame.pack(fill="x", pady=2)
        ttk.Label(time_frame, text="擷取時間(秒):").pack(side="left")
        self.gan_duration = ttk.Entry(time_frame, width=12)
        self.gan_duration.insert(0, "5")
        self.gan_duration.pack(side="left", padx=5)
        self.gan_duration.bind('<KeyRelease>', lambda e: self._update_gan_params('time'))
        
        # 採樣點數
        points_frame = ttk.Frame(lf_params)
        points_frame.pack(fill="x", pady=2)
        ttk.Label(points_frame, text="採樣點數:").pack(side="left")
        self.gan_sample_num = ttk.Entry(points_frame, width=12)
        self.gan_sample_num.insert(0, "2000000")
        self.gan_sample_num.pack(side="left", padx=5)
        self.gan_sample_num.bind('<KeyRelease>', lambda e: self._update_gan_params('points'))
        
        # 提示
        hint = ttk.Label(lf_params, 
                        text="建議:\n短噪音(0.5-1秒): 快速訓練\n長噪音(5-10秒): 更真實", 
                        font=('Arial', 8), foreground="gray", justify="left")
        hint.pack(pady=5)
        
        # 通道配置
        lf_channels_gan = ttk.LabelFrame(left_panel, text="[擷取] 通道配置", padding=10)
        lf_channels_gan.pack(fill="x", pady=5)
        
        # 表頭
        header_frame_gan = ttk.Frame(lf_channels_gan)
        header_frame_gan.pack(fill="x", pady=2)
        ttk.Label(header_frame_gan, text="啟用", width=5).grid(row=0, column=0, padx=2)
        ttk.Label(header_frame_gan, text="通道名稱", width=10).grid(row=0, column=1, padx=2)
        
        # 通道列表
        self.gan_channels = {}
        self.gan_channel_names = {}
        for i, default_ch in enumerate(['ai0', 'ai1', 'ai2', 'ai3']):
            ch_row_gan = ttk.Frame(lf_channels_gan)
            ch_row_gan.pack(fill="x", pady=1)
            
            var = tk.BooleanVar(value=(i < 2))
            cb = ttk.Checkbutton(ch_row_gan, variable=var)
            cb.grid(row=0, column=0, padx=8)
            
            entry = ttk.Entry(ch_row_gan, width=12)
            entry.insert(0, default_ch)
            entry.grid(row=0, column=1, padx=2)
            
            self.gan_channels[default_ch] = var
            self.gan_channel_names[default_ch] = entry
        
        ttk.Label(lf_channels_gan, text="例如: ai0, ai1 或 Dev1/ai0", 
                 font=('Arial', 7), foreground="gray").pack(pady=2)
        
        # 收集模式
        lf_mode = ttk.LabelFrame(left_panel, text="模式", padding=10)
        lf_mode.pack(fill="both", expand=True, pady=5)
        
        # Mode notebook
        mode_nb = ttk.Notebook(lf_mode)
        mode_nb.pack(fill="both", expand=True)
        
        # 單次模式
        single_frame = ttk.Frame(mode_nb)
        mode_nb.add(single_frame, text="單次收集")
        
        self.btn_gan_single_capture = tk.Button(single_frame, text="[擷取] 擷取訊號",
                                                command=self._gan_single_capture,
                                                bg="#4CAF50", fg="white",
                                                font=('Arial', 11, 'bold'),
                                                height=2, cursor="hand2")
        self.btn_gan_single_capture.pack(fill="x", padx=20, pady=10)
        
        self.btn_gan_single_save = tk.Button(single_frame, text="儲存",
                                            command=self._gan_single_save,
                                            state='disabled',
                                            font=('Arial', 10),
                                            height=2, cursor="hand2")
        self.btn_gan_single_save.pack(fill="x", padx=20, pady=5)
        
        # 長時間模式
        long_frame = ttk.Frame(mode_nb)
        mode_nb.add(long_frame, text="長時間收集")
        
        # 時長設定（與標準數據相同）
        dur_frame = ttk.Frame(long_frame)
        dur_frame.pack(fill="x", pady=5)
        ttk.Label(dur_frame, text="時長:").pack(side="left", padx=5)
        self.gan_long_duration = ttk.Entry(dur_frame, width=8)
        self.gan_long_duration.insert(0, "10")
        self.gan_long_duration.pack(side="left")
        self.gan_long_unit = ttk.Combobox(dur_frame, width=6, state="readonly")
        self.gan_long_unit['values'] = ("秒", "分鐘", "小時")
        self.gan_long_unit.current(1)
        self.gan_long_unit.pack(side="left", padx=2)
        
        int_frame = ttk.Frame(long_frame)
        int_frame.pack(fill="x", pady=5)
        ttk.Label(int_frame, text="間隔:").pack(side="left", padx=5)
        self.gan_long_interval = ttk.Entry(int_frame, width=8)
        self.gan_long_interval.insert(0, "5")
        self.gan_long_interval.pack(side="left")
        self.gan_long_interval_unit = ttk.Combobox(int_frame, width=6, state="readonly")
        self.gan_long_interval_unit['values'] = ("秒", "分鐘")
        self.gan_long_interval_unit.current(0)
        self.gan_long_interval_unit.pack(side="left", padx=2)
        
        self.gan_estimate = ttk.Label(long_frame, text="預估: -- 筆", foreground="blue")
        self.gan_estimate.pack(pady=5)
        
        self.btn_gan_long_start = tk.Button(long_frame, text="開始",
                                            command=self._gan_long_start,
                                            bg="#4CAF50", fg="white",
                                            font=('Arial', 11, 'bold'),
                                            height=2, cursor="hand2")
        self.btn_gan_long_start.pack(fill="x", padx=20, pady=5)
        
        self.btn_gan_long_stop = tk.Button(long_frame, text="停止",
                                           command=self._gan_long_stop,
                                           bg="#f44336", fg="white",
                                           font=('Arial', 11, 'bold'),
                                           height=2, cursor="hand2",
                                           state='disabled')
        self.btn_gan_long_stop.pack(fill="x", padx=20, pady=5)
        
        # ===== 右側：記錄區 =====
        
        self.gan_status = ttk.Label(right_panel, text="就緒", 
                                    font=('Arial', 10), foreground="gray")
        self.gan_status.pack(pady=5)
        
        lf_records = ttk.LabelFrame(right_panel, text="記錄", padding=5)
        lf_records.pack(fill="both", expand=True)
        
        self.gan_records_text = scrolledtext.ScrolledText(lf_records, 
                                                          width=60, height=25,
                                                          font=('Courier', 9),
                                                          wrap=tk.WORD)
        self.gan_records_text.pack(fill="both", expand=True)
        
        # 初始化
        self._update_gan_filename()
        self._update_gan_params('time')
        self._update_gan_device_mode()
    
    def _update_gan_device_mode(self):
        """更新GAN設備模式狀態"""
        if self.gan_simulation_mode.get():
            self.gan_device_status.config(
                text="[模擬] 測試模式", 
                foreground="orange")
            self.gan_device_name.config(state='disabled')
        else:
            device_name = self.gan_device_name.get().strip()
            if NIDAQ_AVAILABLE:
                self.gan_device_status.config(
                    text=f"[實際] {device_name}", 
                    foreground="green")
            else:
                self.gan_device_status.config(
                    text=f"NI-DAQ 未載入: {(NIDAQ_IMPORT_ERROR or 'unknown')[:90]}",
                    foreground="red")
            self.gan_device_name.config(state='normal')
    
    def _update_gan_filename(self):
        """更新GAN檔名預覽"""
        try:
            noise_type = self.gan_noise_type.get()
            pressure = self.gan_pressure.get()
            filename = f"{noise_type}_{pressure}bar.tdms"
            self.gan_filename_preview.config(text=filename)
        except:
            self.gan_filename_preview.config(text="(檢查輸入)")
    
    def _update_gan_params(self, changed):
        """更新GAN採樣參數"""
        try:
            rate = float(self.gan_sample_rate.get())
            
            if changed == 'time':
                duration = float(self.gan_duration.get())
                points = int(rate * duration)
                self.gan_sample_num.delete(0, tk.END)
                self.gan_sample_num.insert(0, str(points))
            
            elif changed == 'points':
                points = int(self.gan_sample_num.get())
                duration = points / rate
                self.gan_duration.delete(0, tk.END)
                self.gan_duration.insert(0, f"{duration:.4f}")
        except:
            pass
    
    def _gan_single_capture(self):
        """GAN:單次擷取"""
        self.gan_status.config(text="擷取中...", foreground="blue")
        self.btn_gan_single_capture.config(state='disabled')
        
        def capture():
            try:
                rate = int(self.gan_sample_rate.get())
                points = int(self.gan_sample_num.get())
                
                # 配置設備
                device_name = self.gan_device_name.get().strip()
                simulation_mode = self.gan_simulation_mode.get()
                self.multi_capture.configure_device(device_name, simulation_mode)
                
                # 配置通道（使用可編輯的通道名稱）
                channels = []
                for default_ch, var in self.gan_channels.items():
                    if var.get():  # 如果勾選
                        actual_ch_name = self.gan_channel_names[default_ch].get().strip()
                        if actual_ch_name:  # 如果有輸入名稱
                            channels.append({'id': actual_ch_name, 'enabled': True})
                
                if not channels:
                    self.root.after(0, lambda: self.gan_status.config(
                        text="請至少啟用一個通道", foreground="orange"))
                    return
                
                self.multi_capture.configure_channels(channels)
                
                # 更新狀態
                mode_text = "模擬模式" if simulation_mode else f"設備: {device_name}"
                self.root.after(0, lambda: self.gan_device_status.config(
                    text=f"[擷取] {mode_text}", foreground="blue"))
                
                self.current_gan_data = self.multi_capture.capture_sync(rate, points)
                
                self.root.after(0, self._display_gan_waveform)
                self.root.after(0, lambda: self.gan_status.config(
                    text=f"已擷取 {len(self.current_gan_data)} 個通道", 
                    foreground="green"))
                self.root.after(0, lambda: self.btn_gan_single_save.config(state='normal'))
                
            except Exception as e:
                error_msg = str(e)
                # 簡化錯誤訊息
                if "擷取失敗" in error_msg:
                    display_msg = error_msg.split('\n')[0]
                else:
                    display_msg = f"擷取失敗: {error_msg[:30]}"
                
                self.root.after(0, lambda: self.gan_status.config(
                    text=display_msg, foreground="red"))
            finally:
                self.root.after(0, lambda: self.btn_gan_single_capture.config(state='normal'))
        
        threading.Thread(target=capture, daemon=True).start()
    
    def _display_gan_waveform(self):
        """顯示GAN波形"""
        if not hasattr(self, 'current_gan_data'):
            return
        
        self.gan_records_text.delete(1.0, tk.END)
        self.gan_records_text.insert(tk.END, "=" * 60 + "\n")
        self.gan_records_text.insert(tk.END, "噪音波形預覽\n")
        self.gan_records_text.insert(tk.END, "=" * 60 + "\n\n")
        
        for ch_name, data in self.current_gan_data.items():
            rms = np.sqrt(np.mean(data ** 2))
            peak = np.max(np.abs(data))
            
            self.gan_records_text.insert(tk.END, f"{ch_name}:\n")
            self.gan_records_text.insert(tk.END, f"  RMS:  {rms:.6f} V\n")
            self.gan_records_text.insert(tk.END, f"  Peak: {peak:.6f} V\n")
            self.gan_records_text.insert(tk.END, f"  Length: {len(data)} points\n\n")
        
        self.gan_records_text.insert(tk.END, "\n點擊「儲存」以儲存\n")
    
    def _gan_single_save(self):
        """儲存GAN單次數據"""
        if not hasattr(self, 'current_gan_data'):
            self.gan_status.config(text="沒有數據可儲存", foreground="orange")
            return
        
        try:
            folder = Path(self.gan_folder.get())
            noise_type = self.gan_noise_type.get()
            subfolder = folder / noise_type
            subfolder.mkdir(parents=True, exist_ok=True)
            
            filename = self.gan_filename_preview.cget("text")
            base_name = filename.replace('.tdms', '')
            
            counter = 1
            final_path = subfolder / filename
            
            while final_path.exists():
                filename = f"{base_name}_{counter:03d}.tdms"
                final_path = subfolder / filename
                counter += 1
            
            metadata = {
                'noise_type': noise_type,
                'pressure': self.gan_pressure.get() + 'bar',
                'sample_rate': int(self.gan_sample_rate.get()),
                'sample_num': int(self.gan_sample_num.get()),
                'n_channels': len(self.current_gan_data),
                'sync_mode': True,
                'daq_task': 'single_task_multi_channel' if len(self.current_gan_data) > 1 else 'single_task_single_channel',
                'sensor_map': 'ai0,ai1',
                'timestamp': datetime.now().isoformat()
            }
            
            success = TDMSSaver.save(final_path, self.current_gan_data, metadata)
            
            if success:
                self.gan_records_text.insert(tk.END, f"\n{'='*60}\n")
                self.gan_records_text.insert(tk.END, f"已儲存: {final_path}\n")
                self.gan_status.config(text=f"已儲存: {final_path.name}", foreground="green")
                self.btn_gan_single_save.config(state='disabled')
        
        except Exception as e:
            self.gan_status.config(text=f"儲存失敗: {str(e)[:30]}", foreground="red")
    
    def _gan_long_start(self):
        """GAN長時間收集"""
        self.gan_status.config(text="功能開發中", foreground="orange")
    
    def _gan_long_stop(self):
        """停止GAN長時間收集"""
        pass
    
    # ==================== 通用功能 ====================
    
    def _browse_folder(self, entry_widget):
        """瀏覽資料夾"""
        folder = filedialog.askdirectory(initialdir="./")
        if folder:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, folder)
    
    # ==================== 通用功能 ====================
    
    def _browse_folder(self, entry_widget):
        """瀏覽資料夾"""
        folder = filedialog.askdirectory(initialdir="./")
        if folder:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, folder)


# ============================================================
# 主程式
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("數據收集系統 v17 - 真同步雙通道 transition 3ch 輸出版")
    print("=" * 70)
    print("新功能:")
    print("  1. [OK] 波形圖形化顯示（matplotlib）")
    print("  2. [OK] 即時頻譜分析（FFT + STFT）")
    print("  3. [OK] 收集進度條")
    print("  4. [OK] 完善錯誤處理")
    print("=" * 70)
    print(f"TDMS:   {'OK' if TDMS_AVAILABLE else 'NO (需安裝: pip install npTDMS)'}")
    print(f"NI-DAQ: {'OK' if NIDAQ_AVAILABLE else 'NO (使用模擬模式)'}")
    if not NIDAQ_AVAILABLE:
        print(f"NI-DAQ import error: {NIDAQ_IMPORT_ERROR}")
        print(f"Expected nidaq_capture.py: {NIDAQ_CAPTURE_PATH}")
    print("=" * 70)
    
    root = tk.Tk()
    app = DataCollectionGUI(root)
    root.mainloop()
