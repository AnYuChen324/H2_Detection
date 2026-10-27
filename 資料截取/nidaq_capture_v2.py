# -*- coding: utf-8 -*-
"""
NI-DAQ 擷取模組（NI 9223 / cDAQ）— 對應 data_collection_system_v21.py

提供：
- AEsignal_Capture                     單通道有限擷取（保留舊版介面）
- AEsignal_MultiChannelCapture         多通道同步有限擷取（同一個 task，共用取樣時脈）
- AEsignal_DualChannelContinuousReader 雙通道連續擷取（即時 AE Spectrum 用），含模擬模式
- NIDAQMX_IMPORT_ERROR                 nidaqmx 載入失敗原因；空字串表示可用

舊版（只有 AEsignal_Capture）備份為 nidaq_capture_old_20251212.py
"""

import os
import threading
import time

import numpy as np

try:
    import nidaqmx
    from nidaqmx.constants import (AcquisitionType, Coupling, LoggingMode,
                                   LoggingOperation, READ_ALL_AVAILABLE,
                                   TerminalConfiguration)
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    NIDAQMX_IMPORT_ERROR = ""
except Exception as exc:  # 未安裝 nidaqmx 套件或 NI-DAQmx 驅動
    nidaqmx = None
    NIDAQMX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _require_nidaqmx():
    if nidaqmx is None:
        raise RuntimeError(f"nidaqmx 無法使用：{NIDAQMX_IMPORT_ERROR}")


def _terminal_config(name):
    if not name:
        return TerminalConfiguration.DEFAULT
    mapping = {
        "DIFF": TerminalConfiguration.DIFF,
        "RSE": TerminalConfiguration.RSE,
        "NRSE": TerminalConfiguration.NRSE,
        "PSEUDO_DIFF": TerminalConfiguration.PSEUDO_DIFF,
    }
    key = str(name).strip().upper()
    if key not in mapping:
        raise ValueError(f"不支援的接線模式：{name}")
    return mapping[key]


def _add_voltage_channels(task, channelnames, min_val=-10.0, max_val=10.0,
                          coupling=None, terminal_config=None):
    """在同一個 task 加入所有通道，套用量程 / 接線 / 耦合設定。"""
    term = _terminal_config(terminal_config)
    for name in channelnames:
        chan = task.ai_channels.add_ai_voltage_chan(
            name, terminal_config=term, min_val=min_val, max_val=max_val)
        if coupling:
            want = Coupling.AC if str(coupling).upper().startswith("AC") else Coupling.DC
            try:
                chan.ai_coupling = want
            except Exception as exc:
                raise RuntimeError(
                    f"{name} 不支援 {coupling} 耦合（NI 9223 只支援 DC），請改回 DC。\n{exc}") from exc


#-------------------------------------- Signal Capture with NIDAQ NI9223 -----------------------------------------------------------------------------------
class AEsignal_Capture:
    """
    輸入：取樣數據量、取樣頻率、擷取卡通道名稱、儲存功能開關(0,1)、儲存位置
    """
    def __init__(self, samps, samplerate, channelname, savecheck, savepath):
        self.data_cu = []
        self.time_cu = []
        self.samplesum = samps
        self.samplerate = samplerate
        self.channelname = channelname
        self.savecheck = savecheck
        self.savepath = savepath

    def capture(self):
        _require_nidaqmx()
        datename = time.strftime("%Y%m%d_%H%M%S")

        #擷取執行緒
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.channelname, min_val=-10, max_val=10)
            task.timing.cfg_samp_clk_timing(self.samplerate, sample_mode=AcquisitionType.FINITE, samps_per_chan=self.samplesum)

            #儲存功能
            if self.savecheck == "1":
                try:
                    #路徑判斷
                    if os.path.isabs(self.savepath):
                        task.in_stream.configure_logging(r""+self.savepath+datename+".tdms", LoggingMode.LOG_AND_READ, group_name="Sample", operation=LoggingOperation.OPEN_OR_CREATE)
                    else:
                        task.in_stream.configure_logging(self.savepath+datename+".tdms", LoggingMode.LOG_AND_READ, group_name="Sample", operation=LoggingOperation.OPEN_OR_CREATE)
                except:
                    pass

            timeout = self.samplesum / float(self.samplerate) + 10.0
            data = task.read(READ_ALL_AVAILABLE, timeout=timeout)
            for i in range(len(data)):
                self.data_cu.append(data[i])
                self.time_cu.append(i)
        return self.data_cu, self.time_cu


class AEsignal_MultiChannelCapture:
    """
    多通道同步有限擷取：所有通道放在同一個 task，共用取樣時脈（NI 9223 為同步取樣）。
    capture() 回傳 ({完整通道名: ndarray}, 取樣索引 ndarray)
    """
    def __init__(self, samps, samplerate, channelnames,
                 min_val=-10.0, max_val=10.0, coupling=None, terminal_config=None):
        self.samps = int(samps)
        self.samplerate = float(samplerate)
        self.channelnames = list(channelnames)
        self.min_val = min_val
        self.max_val = max_val
        self.coupling = coupling
        self.terminal_config = terminal_config

    def capture(self):
        _require_nidaqmx()
        if not self.channelnames:
            raise ValueError("沒有指定通道")

        with nidaqmx.Task() as task:
            _add_voltage_channels(task, self.channelnames, self.min_val, self.max_val,
                                  self.coupling, self.terminal_config)
            task.timing.cfg_samp_clk_timing(
                self.samplerate, sample_mode=AcquisitionType.FINITE, samps_per_chan=self.samps)

            data = np.empty((len(self.channelnames), self.samps), dtype=np.float64)
            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()
            timeout = self.samps / self.samplerate + 10.0
            reader.read_many_sample(data, number_of_samples_per_channel=self.samps, timeout=timeout)
            task.stop()

        result = {name: data[i] for i, name in enumerate(self.channelnames)}
        return result, np.arange(self.samps)


class AEsignal_DualChannelContinuousReader:
    """
    雙通道連續擷取。用法：
        reader = AEsignal_DualChannelContinuousReader(samplerate, [ch0, ch1], chunk_size).start()
        chunk = reader.read_chunk()   # {"ai0": ndarray, "ai1": ndarray}，ai0/ai1 依 channelnames 順序
        reader.close()
    simulation=True 時不碰硬體，依取樣率節奏產生雜訊 + 偶發 AE 事件。
    """
    def __init__(self, samplerate, channelnames, chunk_size=8192, simulation=False,
                 min_val=-10.0, max_val=10.0, coupling=None, terminal_config=None,
                 daq_buffer_seconds=2.0):
        self.samplerate = float(samplerate)
        self.channelnames = list(channelnames)
        if len(self.channelnames) != 2:
            raise ValueError(f"需要剛好 2 個通道，收到 {len(self.channelnames)} 個")
        self.chunk_size = int(chunk_size)
        self.simulation = bool(simulation)
        self.min_val = min_val
        self.max_val = max_val
        self.coupling = coupling
        self.terminal_config = terminal_config
        self.daq_buffer_seconds = float(daq_buffer_seconds)

        self._task = None
        self._reader = None
        self._buffer = None
        self._lock = threading.Lock()
        self._closed = False
        self._sim_next_time = None
        self._rng = np.random.default_rng()

    def start(self):
        self._closed = False
        if self.simulation:
            self._sim_next_time = time.perf_counter()
            return self

        _require_nidaqmx()
        task = nidaqmx.Task()
        try:
            _add_voltage_channels(task, self.channelnames, self.min_val, self.max_val,
                                  self.coupling, self.terminal_config)
            buffer_samps = max(self.chunk_size * 4, int(self.samplerate * self.daq_buffer_seconds))
            task.timing.cfg_samp_clk_timing(
                self.samplerate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=buffer_samps)
            task.in_stream.input_buf_size = buffer_samps
            self._reader = AnalogMultiChannelReader(task.in_stream)
            self._buffer = np.empty((2, self.chunk_size), dtype=np.float64)
            task.start()
        except Exception:
            task.close()
            raise
        self._task = task
        return self

    def read_chunk(self):
        if self._closed:
            raise RuntimeError("reader 已關閉")
        if self.simulation:
            return self._simulate_chunk()

        with self._lock:
            if self._task is None:
                raise RuntimeError("reader 尚未啟動")
            timeout = max(10.0, 2.0 * self.chunk_size / self.samplerate)
            self._reader.read_many_sample(
                self._buffer, number_of_samples_per_channel=self.chunk_size, timeout=timeout)
            data = self._buffer.astype(np.float32)
        return {"ai0": data[0], "ai1": data[1]}

    def close(self):
        self._closed = True
        with self._lock:
            task, self._task = self._task, None
            if task is not None:
                try:
                    task.stop()
                finally:
                    task.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def _simulate_chunk(self):
        # 依取樣率節奏輸出，模擬真實擷取的時間
        self._sim_next_time += self.chunk_size / self.samplerate
        delay = self._sim_next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        n = self.chunk_size
        ai0 = self._rng.normal(0.0, 0.001, n)
        ai1 = self._rng.normal(0.0, 0.001, n)
        if n > 2000 and self._rng.random() < 0.3:
            length = min(n // 2, 2000)
            start = int(self._rng.integers(0, n - length))
            t = np.arange(length) / self.samplerate
            freq = 50_000 + self._rng.normal(0, 5_000)
            event = np.exp(-t * 5_000) * np.sin(2 * np.pi * freq * t) * 0.05
            ai0[start:start + length] += event
            ai1[start:start + length] += event * 0.6
        return {"ai0": ai0.astype(np.float32), "ai1": ai1.astype(np.float32)}
