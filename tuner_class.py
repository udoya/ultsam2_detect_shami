"""リアルタイム音程検出チューナー

マイクからの音声入力をリアルタイムで解析し、以下を表示する：
- 音程（音名・オクターブ）
- 音量レベル
- メルスペクトログラムの可視化
- HPS処理による高精度な基本周波数検出
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 音声処理用ライブラリ
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("警告: PyAudioが利用できません。音声機能は無効になります。")
    print("PyAudioをインストールするには: pip install pyaudio")

# 音程解析用ライブラリ
try:
    import librosa
    import librosa.display
    from scipy.signal import butter, lfilter

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("警告: librosaまたはscipyが利用できません。音程検出機能は無効になります。")
    print("ライブラリをインストールするには: pip install librosa scipy")


class HarmonicProductSpectrum:
    """Harmonic Product Spectrum (HPS) 処理クラス - 改良版"""

    def __init__(self, num_harmonics: int = 6, noise_floor: float = 0.01):
        """初期化

        Args:
            num_harmonics: 使用する倍音の数
            noise_floor: ノイズフロア閾値

        """
        self.num_harmonics = num_harmonics
        self.noise_floor = noise_floor

    def process(self, spectrum: np.ndarray) -> np.ndarray:
        """改良されたHPS処理を実行

        Args:
            spectrum: 入力スペクトラム

        Returns:
            HPS処理されたスペクトラム

        """
        if len(spectrum) == 0:
            return spectrum

        # スペクトラムを正規化
        max_val = np.max(spectrum)
        if max_val == 0:
            return spectrum

        normalized_spectrum = spectrum / max_val

        # ノイズフロア以下をより厳しくカット
        normalized_spectrum = np.where(
            normalized_spectrum < self.noise_floor, 0, normalized_spectrum
        )

        # HPS処理前の前処理: 高周波ノイズの抑制
        # 高周波部分（8kHz以上）を段階的に減衰
        freq_bins = len(normalized_spectrum)
        high_freq_start = int(freq_bins * 0.7)  # 全体の70%以降
        if high_freq_start < freq_bins:
            # 指数関数的減衰を適用
            decay_factor = np.exp(-np.linspace(0, 3, freq_bins - high_freq_start))
            normalized_spectrum[high_freq_start:] *= decay_factor

        # HPS処理 - より強力な倍音除去
        hps_spectrum = normalized_spectrum.copy()

        # 重み付きHPS - 高次倍音ほど重みを強く減らす
        for h in range(2, self.num_harmonics + 1):
            downsampled_len = len(normalized_spectrum) // h
            if downsampled_len > 20:  # 最小長制限を厳しく
                # より強い重み減衰
                weight = 1.0 / (h**1.2)  # 指数を1.2に増加

                # ダウンサンプリング
                downsampled = normalized_spectrum[::h][:downsampled_len]

                # より強力な倍音抑制
                hps_spectrum[:downsampled_len] *= downsampled**weight

        # さらなる後処理: スパイク除去
        # 中央値フィルタでスパイクノイズを除去
        if len(hps_spectrum) > 7:
            from scipy.signal import medfilt

            try:
                # 中央値フィルタ適用
                hps_spectrum = medfilt(hps_spectrum, kernel_size=5)
            except ImportError:
                # scipyが使えない場合は簡易版
                kernel_size = 3
                filtered = np.zeros_like(hps_spectrum)
                for i in range(len(hps_spectrum)):
                    start = max(0, i - kernel_size // 2)
                    end = min(len(hps_spectrum), i + kernel_size // 2 + 1)
                    filtered[i] = np.median(hps_spectrum[start:end])
                hps_spectrum = filtered

        # より強い平滑化（ノイズ除去）
        if len(hps_spectrum) > 9:
            # ガウシアンカーネルでより強い平滑化
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # ガウシアン風
            hps_spectrum = np.convolve(hps_spectrum, kernel, mode="same")

        # 動的レンジ圧縮 - 強い信号をさらに抑制
        hps_spectrum = np.power(hps_spectrum, 0.7)  # 0.7乗で圧縮

        # 最終的な正規化
        max_hps = np.max(hps_spectrum)
        if max_hps > 0:
            hps_spectrum = hps_spectrum / max_hps

        return hps_spectrum


class PitchDetector:
    """音程検出クラス（HPS処理付き）"""

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 4096):
        """初期化

        Args:
            sample_rate: サンプリングレート
            buffer_size: バッファサイズ

        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.enabled = LIBROSA_AVAILABLE

        if not self.enabled:
            print("LibrosaまたはScipyが利用できないため、音程検出は無効です")
            return

        # 音程から音名への変換テーブル
        self.note_names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        # 楽器の音域設定 (拡張)
        self.min_freq = 70.0  # さらに低い音域まで対応
        self.max_freq = 2000.0  # 高音域を拡張

        # HPS処理クラス（改良版パラメータ）
        self.hps = HarmonicProductSpectrum(num_harmonics=6, noise_floor=0.01)

        # ピッチ検出の安定化用（履歴サイズ拡張）
        self.pitch_history = deque(maxlen=5)

        # 前処理用フィルタ
        self.preemphasis_factor = 0.97

        # 検出精度向上のための閾値
        self.min_confidence = 0.25  # 最小信頼度を下げる
        self.peak_threshold_ratio = 0.1  # ピーク閾値を緩和

    def detect_pitch(self, audio_data: np.ndarray) -> tuple[float, str, int, float]:
        """改良された音程検出

        Args:
            audio_data: 音声データ (numpy array)

        Returns:
            (frequency, note_name, octave, confidence) のタプル

        """
        if not self.enabled or len(audio_data) == 0:
            return 0.0, "", 0, 0.0

        try:
            # 前処理: プリエンファシス（高周波数成分を強調）
            if len(audio_data) > 1:
                emphasized = np.append(
                    audio_data[0], audio_data[1:] - self.preemphasis_factor * audio_data[:-1]
                )
            else:
                emphasized = audio_data

            # 窓関数を適用（ハニング窓 + ガウシアンテーパリング）
            window = np.hanning(len(emphasized))
            # ガウシアンテーパリングでエッジ効果を減少
            gaussian_taper = np.exp(
                -0.5
                * ((np.arange(len(emphasized)) - len(emphasized) / 2) / (len(emphasized) / 8)) ** 2
            )
            window = window * gaussian_taper
            windowed = emphasized * window

            # 信号の最小レベルチェック
            rms = np.sqrt(np.mean(windowed**2))
            if rms < 1e-8:  # さらに閾値を緩和
                return 0.0, "", 0, 0.0

            # ゼロパディングでFFT解像度を向上
            padded_length = len(windowed) * 2
            windowed_padded = np.pad(windowed, (0, padded_length - len(windowed)), "constant")

            # FFTを実行
            fft = np.fft.rfft(windowed_padded)
            spectrum = np.abs(fft)

            # 周波数軸を作成
            freqs = np.fft.rfftfreq(padded_length, 1 / self.sample_rate)

            # DCオフセット除去
            if len(spectrum) > 1:
                spectrum[0] = 0

            # HPS処理を適用
            hps_spectrum = self.hps.process(spectrum)

            # 楽器の音域内でピークを検索
            valid_indices = np.where(
                (freqs >= self.min_freq) & (freqs <= self.max_freq),
            )[0]

            if len(valid_indices) < 10:  # 最小データ点数チェック
                return 0.0, "", 0, 0.0

            valid_spectrum = hps_spectrum[valid_indices]
            valid_freqs = freqs[valid_indices]

            # 動的閾値計算
            max_amplitude = np.max(valid_spectrum)
            mean_amplitude = np.mean(valid_spectrum)
            std_amplitude = np.std(valid_spectrum)

            # 動的ノイズ閾値を緩和
            noise_threshold = mean_amplitude + self.peak_threshold_ratio * std_amplitude

            # 最小閾値を設定（検出しやすくする）
            min_threshold = max_amplitude * 0.05  # 最大値の5%
            noise_threshold = max(noise_threshold, min_threshold)

            if max_amplitude < noise_threshold:
                return 0.0, "", 0, 0.0

            # ピーク検出（局所最大値）
            peak_indices = []
            for i in range(1, len(valid_spectrum) - 1):
                if (
                    valid_spectrum[i] > valid_spectrum[i - 1]
                    and valid_spectrum[i] > valid_spectrum[i + 1]
                    and valid_spectrum[i] > noise_threshold
                ):
                    peak_indices.append(i)

            if not peak_indices:
                # ピークが見つからない場合は最大値を使用
                max_idx = np.argmax(valid_spectrum)
                fundamental_freq = valid_freqs[max_idx]
            else:
                # 最も強いピークを選択
                peak_amplitudes = valid_spectrum[peak_indices]
                strongest_peak_idx = peak_indices[np.argmax(peak_amplitudes)]
                fundamental_freq = valid_freqs[strongest_peak_idx]

            # 基本周波数の信頼度を計算（改良版）
            signal_power = max_amplitude
            noise_power = mean_amplitude
            snr = signal_power / (noise_power + 1e-10)

            # 信頼度の正規化（0-1）
            confidence = min(np.log10(snr + 1) / 2.0, 1.0)

            # 最小信頼度チェック
            if confidence < self.min_confidence:
                return 0.0, "", 0, 0.0

            # 履歴を使って安定化
            self.pitch_history.append(fundamental_freq)

            # 履歴の統計的処理
            if len(self.pitch_history) >= 3:
                history_array = np.array(list(self.pitch_history))
                # 外れ値除去のためのトリム平均
                sorted_history = np.sort(history_array)
                trim_size = len(sorted_history) // 4  # 25%をトリム
                if trim_size > 0:
                    trimmed = sorted_history[trim_size:-trim_size]
                    stable_freq = np.mean(trimmed)
                else:
                    stable_freq = np.median(history_array)
            else:
                stable_freq = fundamental_freq

            # 周波数から音名とオクターブを計算
            note_name, octave = self.freq_to_note(stable_freq)

            return stable_freq, note_name, octave, confidence

        except Exception as e:
            print(f"音程検出エラー: {e}")
            return 0.0, "", 0, 0.0

    def freq_to_note(self, frequency: float) -> tuple[str, int]:
        """周波数から音名とオクターブを計算

        Args:
            frequency: 周波数 (Hz)

        Returns:
            (note_name, octave) のタプル

        """
        if frequency <= 0:
            return "", 0

        # A4 = 440Hz を基準とした計算
        A4 = 440.0
        C0 = A4 * np.power(2, -4.75)  # C0の周波数

        if frequency < C0:
            return "", 0

        # セント値を計算
        h = 12 * np.log2(frequency / C0)
        octave = int(h // 12)
        n = int(h % 12)

        return self.note_names[n], octave


class MelSpectrogramProcessor:
    """メルスペクトログラム処理クラス"""

    def __init__(self, sample_rate: int = 44100, n_mels: int = 128, n_fft: int = 2048):
        """初期化

        Args:
            sample_rate: サンプリングレート
            n_mels: メルフィルタバンクの数
            n_fft: FFTサイズ

        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.enabled = LIBROSA_AVAILABLE

        if not self.enabled:
            return

        # メルスペクトログラムの履歴（時間軸表示用）
        self.history_size = 100
        self.mel_history = deque(maxlen=self.history_size)

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """改良されたメルスペクトログラムを計算

        Args:
            audio_data: 音声データ

        Returns:
            メルスペクトログラム (dB)

        """
        if not self.enabled or len(audio_data) == 0:
            return np.zeros((self.n_mels, 1))

        try:
            # 信号レベルチェック
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 1e-8:  # さらに非常に小さい信号も検出
                # 無音状態として処理
                mel_frame = np.full(self.n_mels, -80.0)  # -80dB
                self.mel_history.append(mel_frame)
                return np.zeros((self.n_mels, 1))

            # 前処理: オーディオデータの前処理強化
            # DCオフセット除去
            audio_processed = audio_data - np.mean(audio_data)

            # プリエンファシス（高周波成分強調）を弱めに設定
            preemphasis = 0.95
            if len(audio_processed) > 1:
                audio_processed = np.append(
                    audio_processed[0], audio_processed[1:] - preemphasis * audio_processed[:-1]
                )

            # メルスペクトログラムを計算（パラメータをさらに調整）
            mel_spec = librosa.feature.melspectrogram(
                y=audio_processed,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.n_fft // 8,  # より細かい時間解像度
                window="hann",
                fmin=60.0,  # より低い周波数からカバー
                fmax=6000.0,  # 高周波数をさらに制限
                power=1.5,  # パワーを1.5に調整（2.0より弱く）
                center=True,
                pad_mode="reflect",
            )

            # dBスケールに変換（より保守的な設定）
            mel_spec_db = librosa.power_to_db(
                mel_spec,
                ref=np.max,
                top_db=60.0,  # ダイナミックレンジをさらに制限
                amin=1e-12,  # 最小値をより小さく
            )

            # 時間軸の統計処理（ノイズ減少強化）
            if mel_spec_db.shape[1] > 1:
                # パーセンタイル処理でノイズ除去
                mel_frame = np.percentile(mel_spec_db, 25, axis=1)  # 25パーセンタイル使用
            else:
                mel_frame = mel_spec_db.flatten()

            # 強力なノイズゲート適用
            noise_floor = np.percentile(mel_frame, 10)  # 10パーセンタイルをノイズフロアとする
            mel_frame = np.where(mel_frame < noise_floor + 20, noise_floor, mel_frame)

            # 時間方向の平滑化を強化
            if len(self.mel_history) > 0:
                prev_frame = self.mel_history[-1]
                # より強い平滑化
                alpha = 0.5  # 現在フレームの重みを下げる
                mel_frame = alpha * mel_frame + (1 - alpha) * prev_frame

                # さらに前々フレームとも平均化
                if len(self.mel_history) > 1:
                    prev_prev_frame = self.mel_history[-2]
                    mel_frame = 0.6 * mel_frame + 0.3 * prev_frame + 0.1 * prev_prev_frame

            # 周波数方向の平滑化も追加
            if len(mel_frame) > 5:
                # ガウシアンフィルタ風の平滑化
                kernel = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
                mel_frame = np.convolve(mel_frame, kernel, mode="same")

            # 異常値の除去
            median_val = np.median(mel_frame)
            mad = np.median(np.abs(mel_frame - median_val))
            threshold = median_val + 3 * mad
            mel_frame = np.where(mel_frame > threshold, threshold, mel_frame)

            # 履歴に追加
            self.mel_history.append(mel_frame)

            return mel_spec_db

        except Exception as e:
            print(f"メルスペクトログラム処理エラー: {e}")
            return np.zeros((self.n_mels, 1))

    def get_history_matrix(self) -> np.ndarray:
        """履歴からマトリックス形式のメルスペクトログラムを取得

        Returns:
            メルスペクトログラムマトリックス (n_mels, history_size)

        """
        if not self.mel_history:
            return np.zeros((self.n_mels, self.history_size))

        # 履歴をマトリックスに変換
        history_matrix = np.array(list(self.mel_history)).T

        # サイズを調整
        if history_matrix.shape[1] < self.history_size:
            # 不足分をゼロパディング
            padding = np.zeros((self.n_mels, self.history_size - history_matrix.shape[1]))
            history_matrix = np.hstack([padding, history_matrix])

        return history_matrix


class VolumeAnalyzer:
    """音量解析クラス"""

    def __init__(self, history_size: int = 50):
        """初期化

        Args:
            history_size: 音量履歴のサイズ

        """
        self.history_size = history_size
        self.volume_history = deque(maxlen=history_size)

    def analyze(self, audio_data: np.ndarray) -> tuple[float, float]:
        """音量を解析

        Args:
            audio_data: 音声データ

        Returns:
            (current_volume, max_volume) のタプル

        """
        if len(audio_data) == 0:
            return 0.0, 0.0

        # RMS音量を計算
        rms_volume = np.sqrt(np.mean(audio_data**2))

        # dBスケールに変換
        if rms_volume > 0:
            db_volume = 20 * np.log10(rms_volume)
            # -60dB to 0dBの範囲で正規化
            normalized_volume = max(0.0, (db_volume + 60) / 60.0)
        else:
            normalized_volume = 0.0

        # 履歴に追加
        self.volume_history.append(normalized_volume)

        # 最大音量を計算
        max_volume = max(self.volume_history) if self.volume_history else 0.0

        return normalized_volume, max_volume


class RealtimeTuner:
    """リアルタイムチューナークラス"""

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 8192,  # バッファサイズを大きくして安定性向上
        input_device_index: int | None = None,
        output_device_index: int | None = None,
        enable_passthrough: bool = True,
    ):
        """初期化

        Args:
            sample_rate: サンプリングレート
            chunk_size: チャンクサイズ
            input_device_index: 入力デバイスのインデックス
            output_device_index: 出力デバイスのインデックス
            enable_passthrough: 音声パススルーを有効にするか

        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.enable_passthrough = enable_passthrough
        self.enabled = PYAUDIO_AVAILABLE and LIBROSA_AVAILABLE

        if not self.enabled:
            print("必要なライブラリが利用できないため、チューナーは無効です")
            return

        # 音声処理コンポーネント
        self.pitch_detector = PitchDetector(sample_rate, chunk_size)
        self.mel_processor = MelSpectrogramProcessor(sample_rate)
        self.volume_analyzer = VolumeAnalyzer()

        # PyAudioの初期化
        try:
            self.audio = pyaudio.PyAudio()
        except Exception as e:
            print(f"PyAudioの初期化に失敗しました: {e}")
            self.enabled = False
            return

        # 音声データキュー
        self.audio_queue = queue.Queue(maxsize=10)
        self.passthrough_queue = queue.Queue(maxsize=5)  # パススルー用キュー
        self.stop_event = threading.Event()

        # データ同期用ロック
        self.data_lock = threading.Lock()
        self.data_updated = threading.Event()

        # 分析結果
        self.current_pitch = {"frequency": 0.0, "note": "", "octave": 0, "confidence": 0.0}
        self.current_volume = {"level": 0.0, "max": 0.0}
        self.current_mel = np.zeros((128, 100))

        # ストリーム
        self.input_stream = None
        self.output_stream = None

    def list_audio_devices(self):
        """利用可能な音声デバイスを一覧表示"""
        if not self.enabled:
            return

        device_count = self.audio.get_device_count()
        print("利用可能な音声デバイス:")

        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            print(
                f"  {i}: {device_info['name']} (入力: {device_info['maxInputChannels']}, 出力: {device_info['maxOutputChannels']})",
            )

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音声入力コールバック"""
        if status:
            print(f"音声入力エラー: {status}")

        try:
            # 分析用キューに音声データを追加
            self.audio_queue.put_nowait(in_data)
        except queue.Full:
            # キューが満杯の場合は古いデータを削除
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(in_data)
            except queue.Empty:
                pass

        # パススルー用キューにも追加
        if self.enable_passthrough:
            try:
                self.passthrough_queue.put_nowait(in_data)
            except queue.Full:
                try:
                    self.passthrough_queue.get_nowait()
                    self.passthrough_queue.put_nowait(in_data)
                except queue.Empty:
                    pass

        return (None, pyaudio.paContinue)

    def _output_callback(self, in_data, frame_count, time_info, status):
        """音声出力コールバック（パススルー用）"""
        if status:
            print(f"音声出力エラー: {status}")

        try:
            # パススルーキューから音声データを取得
            audio_data = self.passthrough_queue.get_nowait()
            return (audio_data, pyaudio.paContinue)
        except queue.Empty:
            # データがない場合は無音を出力
            silence = b"\x00" * (frame_count * 4)  # float32のため4バイト
            return (silence, pyaudio.paContinue)

    def _process_audio_thread(self):
        """音声処理スレッド"""
        while not self.stop_event.is_set():
            try:
                # キューから音声データを取得(タイムアウト付き)
                audio_data = self.audio_queue.get(timeout=0.1)

                # バイトデータをnumpy配列に変換
                audio_array = np.frombuffer(audio_data, dtype=np.float32)

                # 音程検出
                frequency, note, octave, confidence = self.pitch_detector.detect_pitch(audio_array)
                pitch_result = {
                    "frequency": frequency,
                    "note": note,
                    "octave": octave,
                    "confidence": confidence,
                    "note_display": f"{note}{octave}" if note else "",
                }

                # 音量解析
                volume_level, max_volume = self.volume_analyzer.analyze(audio_array)
                volume_result = {
                    "level": volume_level,
                    "max": max_volume,
                }

                # メルスペクトログラム処理
                self.mel_processor.process(audio_array)
                mel_result = self.mel_processor.get_history_matrix()

                # データを安全に更新
                with self.data_lock:
                    self.current_pitch = pitch_result
                    self.current_volume = volume_result
                    self.current_mel = mel_result
                    self.data_updated.set()  # 更新フラグを設定

            except queue.Empty:
                continue
            except Exception as e:
                print(f"音声処理エラー: {e}")

    def start_analysis(self):
        """音声解析を開始"""
        if not self.enabled:
            print("チューナーが無効のため、解析を開始できません")
            return False

        try:
            # 入力ストリームを開始
            self.input_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )

            # パススルー用出力ストリームを開始
            if self.enable_passthrough:
                self.output_stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self.output_device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._output_callback,
                )
                print("音声パススルーを開始しました")

            # 音声処理スレッドを開始
            self.processing_thread = threading.Thread(target=self._process_audio_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            print("音声解析を開始しました")
            return True

        except Exception as e:
            print(f"音声解析の開始に失敗しました: {e}")
            return False

    def stop_analysis(self):
        """音声解析を停止"""
        if not self.enabled:
            return

        self.stop_event.set()

        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()

        print("音声解析を停止しました")

    def get_pitch_info(self) -> dict:
        """現在の音程情報を取得"""
        return self.current_pitch.copy()

    def get_volume_info(self) -> dict:
        """現在の音量情報を取得"""
        return self.current_volume.copy()

    def get_mel_spectrogram(self) -> np.ndarray:
        """現在のメルスペクトログラムを取得"""
        return self.current_mel.copy()

    def draw_tuner_display(self, width: int = 800, height: int = 600) -> np.ndarray:
        """チューナー表示画面を描画

        Args:
            width: 画面幅
            height: 画面高さ

        Returns:
            描画された画像

        """
        # 黒い背景を作成
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # データを安全にコピー
        with self.data_lock:
            pitch_data = self.current_pitch.copy()
            volume_data = self.current_volume.copy()
            mel_data = self.current_mel.copy()

        # 音程表示エリア (上部1/3)
        self._draw_pitch_display(image, 0, 0, width, height // 3, pitch_data)

        # 音量表示エリア (中央1/3)
        self._draw_volume_display(image, 0, height // 3, width, height // 3, volume_data)

        # メルスペクトログラム表示エリア (下部1/3)
        self._draw_mel_display(image, 0, 2 * height // 3, width, height // 3, mel_data)

        return image

    def _draw_pitch_display(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        pitch_info: dict,
    ):
        """音程表示を描画"""
        # 背景
        cv2.rectangle(image, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (100, 100, 100), 2)

        # タイトル
        cv2.putText(
            image,
            "PITCH DETECTION",
            (x + 20, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 音名表示
        note_display = pitch_info.get("note_display", "")
        if note_display:
            # 大きく音名を表示
            note_size = cv2.getTextSize(note_display, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 3)[0]
            note_x = x + (width - note_size[0]) // 2
            note_y = y + height // 2

            cv2.putText(
                image,
                note_display,
                (note_x, note_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0,
                (0, 255, 255),  # 黄色
                3,
            )

            # 周波数表示
            freq_text = f"{pitch_info.get('frequency', 0):.1f} Hz"
            freq_size = cv2.getTextSize(freq_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            freq_x = x + (width - freq_size[0]) // 2
            freq_y = note_y + 50

            cv2.putText(
                image,
                freq_text,
                (freq_x, freq_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 200),  # 薄い黄色
                2,
            )

            # 信頼度表示
            confidence = pitch_info.get("confidence", 0)
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(
                image,
                conf_text,
                (x + 20, y + height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if confidence > 0.5 else (0, 100, 255),
                2,
            )
        else:
            # 音が検出されていない場合
            cv2.putText(
                image,
                "No Signal",
                (x + width // 2 - 80, y + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (100, 100, 100),
                2,
            )

    def _draw_volume_display(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        volume_info: dict,
    ):
        """音量表示を描画"""
        # 背景
        cv2.rectangle(image, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (100, 100, 100), 2)

        # タイトル
        cv2.putText(
            image,
            "VOLUME LEVEL",
            (x + 20, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 音量バー
        bar_x = x + 50
        bar_y = y + 50
        bar_width = width - 100
        bar_height = 40

        # 背景バー
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1,
        )
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (100, 100, 100),
            2,
        )

        # 現在の音量バー
        current_level = volume_info.get("level", 0)
        current_width = int(bar_width * current_level)

        if current_width > 0:
            # 音量レベルに応じて色を変更
            if current_level < 0.3:
                color = (0, 255, 0)  # 緑
            elif current_level < 0.7:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 0, 255)  # 赤

            cv2.rectangle(
                image,
                (bar_x, bar_y),
                (bar_x + current_width, bar_y + bar_height),
                color,
                -1,
            )

        # 最大音量マーカー
        max_level = volume_info.get("max", 0)
        max_x = bar_x + int(bar_width * max_level)
        cv2.line(image, (max_x, bar_y), (max_x, bar_y + bar_height), (255, 255, 255), 2)

        # 数値表示
        level_text = f"Level: {current_level:.2f}"
        cv2.putText(
            image,
            level_text,
            (x + 50, y + height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        max_text = f"Max: {max_level:.2f}"
        cv2.putText(
            image,
            max_text,
            (x + 250, y + height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

    def _draw_mel_display(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        mel_data: np.ndarray,
    ):
        """メルスペクトログラム表示を描画"""
        # 背景
        cv2.rectangle(image, (x, y), (x + width, y + height), (20, 20, 20), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (100, 100, 100), 2)

        # タイトル
        cv2.putText(
            image,
            "MEL SPECTROGRAM",
            (x + 20, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # メルスペクトログラム描画エリア
        mel_x = x + 20
        mel_y = y + 40
        mel_width = width - 40
        mel_height = height - 60

        if mel_data.size > 0:
            # データを正規化 (-80dB to 0dB -> 0 to 255)
            normalized = np.clip((mel_data + 80) / 80 * 255, 0, 255).astype(np.uint8)

            # 上下反転（高周波数が上に来るように）
            normalized = np.flipud(normalized)

            # リサイズして表示
            resized = cv2.resize(normalized, (mel_width, mel_height))

            # カラーマップを適用（ホットカラーマップ風）
            colored = cv2.applyColorMap(resized, cv2.COLORMAP_HOT)

            # 画像に描画
            image[mel_y : mel_y + mel_height, mel_x : mel_x + mel_width] = colored

        # 周波数ラベル
        freq_labels = ["8kHz", "4kHz", "2kHz", "1kHz", "500Hz", "250Hz"]
        for i, label in enumerate(freq_labels):
            label_y = mel_y + (i * mel_height // len(freq_labels))
            cv2.putText(
                image,
                label,
                (mel_x + mel_width + 5, label_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

    def __del__(self):
        """デストラクタ"""
        if hasattr(self, "audio") and self.audio:
            self.audio.terminate()


def main():
    """メイン関数"""
    print("リアルタイムチューナー（音声パススルー付き）")
    print("=" * 50)

    # チューナーの初期化（パススルー有効）
    tuner = RealtimeTuner(enable_passthrough=True)

    if not tuner.enabled:
        print("チューナーを初期化できませんでした")
        return

    # 利用可能なデバイスを表示
    tuner.list_audio_devices()

    # 音声解析を開始
    if not tuner.start_analysis():
        print("音声解析の開始に失敗しました")
        return

    print("\n機能:")
    print("  - リアルタイム音程検出（HPS処理付き）")
    print("  - 音量レベル表示")
    print("  - メルスペクトログラム可視化")
    print("  - マイク音声のリアルタイムパススルー")
    print("\n操作:")
    print("  'q' または ESC: 終了")
    print("  's': スクリーンショット保存")

    # 表示ループ
    cv2.namedWindow("Realtime Tuner", cv2.WINDOW_AUTOSIZE)
    fps_counter = 0
    last_fps_time = time.time()

    try:
        while True:
            # データが更新されるまで短時間待機
            if tuner.data_updated.wait(timeout=0.1):
                tuner.data_updated.clear()  # フラグをクリア

                # チューナー表示を生成
                display_image = tuner.draw_tuner_display(1200, 800)

                # FPS計算
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    print(f"Display FPS: {fps:.1f}", end="\r")
                    fps_counter = 0
                    last_fps_time = current_time

                # 画面に表示
                cv2.imshow("Realtime Tuner", display_image)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' または ESC
                break
            if key == ord("s"):  # スクリーンショット
                timestamp = int(time.time())
                filename = f"tuner_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, tuner.draw_tuner_display(1200, 800))
                print(f"\nスクリーンショットを保存しました: {filename}")

    except KeyboardInterrupt:
        print("\nキーボード割り込みで終了しました")

    finally:
        # クリーンアップ
        tuner.stop_analysis()
        cv2.destroyAllWindows()
        print("チューナーを終了しました")


if __name__ == "__main__":
    main()
