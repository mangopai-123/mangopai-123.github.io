import os
import numpy as np
from pydub import AudioSegment
from typing import Union, Dict, Any

class AudioNormalizer:
    def __init__(self, 
                 target_sr: int = 24000, 
                 target_dBFS: float = -20.0, 
                 gain_tolerance: float = 3.0,
                 max_amplitude_ratio: float = 0.8
                ):
        self.target_sr = target_sr
        self.target_dBFS = target_dBFS
        self.min_dBFS = target_dBFS - gain_tolerance
        self.max_dBFS = target_dBFS + gain_tolerance
        self.max_amplitude = 32767 * max_amplitude_ratio 
        self.min_amplitude = -self.max_amplitude

    def _calculate_max_safe_gain(self, samples: np.ndarray) -> float:
 
        if samples.size == 0:
            return 0.0
        
        max_abs = np.max(np.abs(samples))
        if max_abs == 0:
            return 0.0  # 静音，任意增益都安全
        
        # 最大允许放大倍数
        max_gain_factor = self.max_amplitude / max_abs
        max_safe_gain_db = 20 * np.log10(max_gain_factor)
        return max_safe_gain_db

    def standardization(self, audio: Union[str, AudioSegment]) -> Union[Dict[str, Any], None]:
        try:
            if isinstance(audio, str):
                name = os.path.basename(audio)
                seg = AudioSegment.from_file(audio)
            elif isinstance(audio, AudioSegment):
                seg = audio
                name = "in_memory_audio"
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio)}")

            seg = seg.set_frame_rate(self.target_sr)
            seg = seg.set_channels(1)
            seg = seg.set_sample_width(2)

            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= np.iinfo(np.int16).max

            current_dBFS = seg.dBFS

            ideal_gain = self.target_dBFS - current_dBFS

            max_safe_gain = self._calculate_max_safe_gain(samples)
            # actual_gain = min(ideal_gain, max_safe_gain)
            actual_gain = np.clip(ideal_gain, -60, max_safe_gain)  # 限制极端增益


            gain_factor = 10 ** (actual_gain / 20.0)
            new_samples = samples * gain_factor

            # if np.any(new_samples > self.max_amplitude) or np.any(new_samples < self.min_amplitude):
            #     new_samples = np.clip(new_samples, self.min_amplitude, self.max_amplitude)

            new_samples = np.clip(new_samples, -1.0, 1.0)

            new_seg = AudioSegment(
                new_samples.astype(np.int16).tobytes(),
                frame_rate=self.target_sr,
                sample_width=2,
                channels=1
            )
            final_dBFS = new_seg.dBFS

            # if not (self.min_dBFS <= final_dBFS <= self.max_dBFS):
            #     print(f"⚠️ 音频 {name} 最终响度 {final_dBFS:.2f} dBFS 超出目标范围 [{self.min_dBFS}, {self.max_dBFS}]")
            
            return {
                "waveform": new_samples.astype(np.float32),
                "sample_rate": self.target_sr,
                "name": name,
            }

        except Exception as e:
            print(f"[AudioNormalizer] Error processing {audio}: {e}")
            return None

if __name__ == "__main__":
    from scipy.io.wavfile import write

    test_audio = "input/audio_000010.wav"
    normalizer = AudioNormalizer(target_sr=24000)
    result = normalizer.standardization(test_audio)

    if result:
        print("✅ 标准化完成")
        print(f"文件名: {result['name']}")
        print(f"采样率: {result['sample_rate']}")
        print(f"波形振幅范围: {result['waveform'].min():.0f} ~ {result['waveform'].max():.0f}（安全范围：{normalizer.min_amplitude:.0f} ~ {normalizer.max_amplitude:.0f}）")

        # 保存标准化后的wav
        out_path = "std.wav"
        write(out_path, result["sample_rate"], result["waveform"].astype(np.int16))
        print(f"已保存: {out_path}")
    else:
        print("❌ 音频标准化失败")
