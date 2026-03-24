import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tools.demucs.api import Separator  # ✅ 你自己的 Demucs API
import torch
from demucs.htdemucs import HTDemucs

"""
torch.serialization.add_safe_globals([HTDemucs])

from fractions import Fraction
import torch.serialization

torch.serialization.add_safe_globals([Fraction])
"""

class VocalSeparator:
    def __init__(
        self,
        model_id: str = "955717e8",
        repo_dir: str = "./pretrain_model/hub/checkpoints",
        device: str = "cpu",
    ):
        self.device_demucs = device
        self.model_id = model_id
        self.repo_dir = Path(repo_dir)

        #try:
        if True:
            self.separator = Separator(
                model=self.model_id,
                repo=self.repo_dir,
                device=self.device_demucs,
            )
            print(f"[VocalSeparator] Model loaded ({model_id}) on {device}")
        #except Exception as e:
        else:
            print(f"[VocalSeparator] Model init failed: {e}")
            self.separator = None

    def source_separation(self, wav: np.ndarray, sr: int) -> Tuple[Optional[np.ndarray], int]:


        if self.separator is None:
            print("[VocalSeparator] Model not initialized.")
            return None, sr

        try:
            wav_tensor = torch.from_numpy(wav).float().to(self.device_demucs)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)  # [1, T]

            _, separated = self.separator.separate_tensor(wav_tensor, sr)

            if "vocals" not in separated:
                print("[VocalSeparator] Warning: 'vocals' not found in separated output.")
                return None, self.separator.samplerate

            vocals = separated["vocals"]
            if vocals.shape[0] > 1:
                vocals = torch.mean(vocals, dim=0, keepdim=True)

            vocals = torchaudio.functional.resample(vocals, 44100, 24000)
            vocals = vocals.squeeze().cpu().numpy().astype(np.float32)
            return vocals, 24000

        except Exception as e:
            print(f"[VocalSeparator] Error during separation: {e}")
            return None, sr


if __name__ == "__main__":
    import torchaudio
    import soundfile as sf

    wav_path = "./std.wav"
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0).numpy()  # 转单声道0
    # print(waveform)

    # 2. 初始化分离器（CPU版本）
    separator = VocalSeparator(
        model_id="955717e8",
        repo_dir="./pretrain_model/hub/checkpoints",
        device="cuda:0"
    )

    # 3. 执行人声分离
    vocals, sep_sr = separator.source_separation(waveform, sr)

    # 4. 保存结果
    if vocals is not None:
        out_path = "vocals.wav"
        sf.write(out_path, vocals, sep_sr)
        print(f"✅ 人声分离完成: {out_path}, 采样率 {sep_sr}")
    else:
        print("❌ 分离失败或无 'vocals' 轨道")

