# -*- coding: utf-8 -*-
"""
NI-DAQ 擷取模組（NI 9223 / cDAQ）— 對應 data_collection_system_v21.py

提供：
- AEsignal_Capture                     單通道有限擷取（保留舊版介面）
- AEsignal_MultiChannelCapture         多通道同步有限擷取（同一個 task，共用取樣時脈）
- AEsignal_DualChannelContinuousReader 雙通道連續擷取（即時 AE Spectrum 用），含模擬模式
- NIDAQMX_IMPORT_ERROR                 nidaqmx 載入失敗原因；空字串表示可用
- ContinuousChunkRecorder             連續錄製：一個 task 從頭收到尾，每支檔案首尾相接（長時間收集用）

舊版（只有 AEsignal_Capture）備份為 nidaq_capture_old_20251212.py
"""

import os
import threading
import time

import numpy as np
import queue

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
        
        # 連續擷取的時間軸（長時間收集用）：
        #   start_wall_time         = task 啟動（第 0 個取樣點）的電腦時間 time.time()
        #   last_chunk_first_sample = 最近一次 read_chunk() 第一個點的取樣序號（從 0 起算）
        # 某個點的時間 = start_wall_time + 取樣序號 / samplerate，不受電腦忙不忙影響
        self._sample_index = 0
        self.start_wall_time = None
        self.last_chunk_first_sample = None

    def start(self):
        self._closed = False
        self._sample_index = 0
        if self.simulation:
            self._sim_next_time = time.perf_counter()
            self.start_wall_time = time.time()
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
            self.start_wall_time = time.time()   # 啟動後立刻記下
        except Exception:
            task.close()
            raise
        self._task = task
        return self

    def read_chunk(self):
        if self._closed:
            raise RuntimeError("reader 已關閉")
        # 記下這一段第一個點的取樣序號；task 一直在收，所以序號一路接下去
        self.last_chunk_first_sample = self._sample_index
        if self.simulation:
            chunk = self._simulate_chunk()
            self._sample_index += self.chunk_size
            return chunk

        with self._lock:
            if self._task is None:
                raise RuntimeError("reader 尚未啟動")
            timeout = max(10.0, 2.0 * self.chunk_size / self.samplerate)
            self._reader.read_many_sample(
                self._buffer, number_of_samples_per_channel=self.chunk_size, timeout=timeout)
            data = self._buffer.astype(np.float32)
            self._sample_index += self.chunk_size
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
    
    
class ContinuousChunkRecorder:
    """
    連續錄製：一個 DAQ 任務從頭收到尾，每 chunk_size 點交給 on_chunk 存成一支檔案。

    兩個執行緒分工：
      - 讀取（run() 所在的執行緒）：只負責從 DAQ 讀資料、丟進佇列。
        DAQ 驅動在背景持續取樣，讀取只是把緩衝區裡的資料拿出來，所以不會漏。
      - 存檔（內部執行緒）：從佇列拿出來，呼叫 on_chunk 存檔。存檔慢一點不影響取樣。

    不會「默默漏資料」：
      - 存檔長時間跟不上（佇列超過 max_queue_chunks）→ 停止並報錯
      - DAQ 緩衝區溢位（讀太慢）→ nidaqmx 會丟例外 → 停止並報錯
      要嘛完整連續，要嘛明確中斷，不會出現看不見的空檔。

    on_chunk(chunk_number, first_sample_index, data_dict)
      chunk_number       : 第幾支（從 0 起算）
      first_sample_index : 這支第一個點的取樣序號，時間 = reader.start_wall_time + 序號 / 取樣率
      data_dict          : {"ai0": ndarray, "ai1": ndarray}
    """

    def __init__(self, reader, n_chunks, on_chunk, should_continue=None, max_queue_chunks=20):
        self.reader = reader                      # AEsignal_DualChannelContinuousReader
        self.n_chunks = int(n_chunks)             # 總共要收幾支檔案
        self.on_chunk = on_chunk                  # 每收到一支要做什麼（存檔）
        self.should_continue = should_continue or (lambda: True)   # 按下「停止」時回傳 False
        self.max_queue_chunks = int(max_queue_chunks)              # 最多累積幾支沒存的（20 支 = 10 秒）
        self.n_read = 0                           # 已讀幾支
        self.n_written = 0                        # 已存幾支
        self.error = None                         # 出錯時記下原因

    def run(self):
        """阻塞執行到收滿 n_chunks、should_continue() 變 False，或出錯為止。回傳摘要 dict。"""
        q = queue.Queue()                 # 讀取和存檔之間的佇列
        stop_event = threading.Event()    # 存檔出錯時通知讀取停下

        # --- 存檔執行緒：從佇列拿出來存 ---
        def writer():
            while True:
                item = q.get()
                if item is None:          # 收到 None 代表結束
                    return
                if self.error is not None:
                    continue              # 已出錯：清空佇列，不再存
                try:
                    self.on_chunk(*item)
                    self.n_written += 1
                except Exception as exc:  # noqa: BLE001
                    self.error = exc
                    stop_event.set()

        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        # --- 讀取（在目前這個執行緒）：只讀、丟進佇列 ---
        try:
            self.reader.start()           # 只啟動一次 DAQ 任務
            for chunk_number in range(self.n_chunks):
                if stop_event.is_set() or not self.should_continue():
                    break
                data = self.reader.read_chunk()
                first = int(self.reader.last_chunk_first_sample)
                q.put((chunk_number, first, data))
                self.n_read += 1
                if q.qsize() > self.max_queue_chunks:
                    raise RuntimeError(
                        f"存檔跟不上取樣：佇列已累積 {q.qsize()} 支未存檔案，"
                        "為避免記憶體爆掉而停止（請改用較快的硬碟或減少畫面更新）")
        except Exception as exc:          # noqa: BLE001
            if self.error is None:
                self.error = exc
        finally:
            try:
                self.reader.close()       # 結束時才關閉 DAQ 任務
            finally:
                q.put(None)               # 通知存檔執行緒結束
                writer_thread.join()      # 等它把佇列裡剩下的存完

        return {
            "n_read": self.n_read,
            "n_written": self.n_written,
            "error": self.error,
            "start_wall_time": getattr(self.reader, "start_wall_time", None),
        }
