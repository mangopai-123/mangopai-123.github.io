import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from pathlib import Path
from typing import Union, Optional
import onnxruntime as ort


class DenoiserONNX:

    def __init__(
        self,
        model_path: Union[str, Path] = "./pretrain_model/denoiser_model.onnx",
        enabled: bool = True,
        device: str = "cpu",
        target_sr: int = 48000,
        atten_lim_db: float = 25.0,
        hop_size: int = 480,
        fft_size: int = 960,
    ):
       
        self.enabled = enabled
        self.device = device
        self.target_sr = target_sr
        self.atten_lim_db = np.array([atten_lim_db], dtype=np.float32)
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.model_path = Path(model_path)

        if not enabled:
            print("[DenoiserONNX] 降噪功能关闭")
            self.session = None
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {self.model_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        #sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # 或 ORT_ENABLE_BASIC
        #sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC    # 或 ORT_ENABLE_BASIC
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )
        print(f"[DenoiserONNX] Model loaded ({self.model_path}) on {device}")

  
    def _pad_and_chunk(self, audio: torch.Tensor) -> tuple:
        orig_len = audio.shape[0]
        hop_pad = (self.hop_size - orig_len % self.hop_size) % self.hop_size
        padded_len = orig_len + hop_pad
        audio_padded = F.pad(audio, (0, self.fft_size + hop_pad))
        chunks = torch.split(audio_padded, self.hop_size)
        return chunks, orig_len, padded_len

    def _denoise_chunks(self, chunks: tuple) -> torch.Tensor:
        state = np.zeros(45304, dtype=np.float32)
        enhanced_chunks = []

        for chunk in chunks:
            out = self.session.run(
                None,
                input_feed={
                    "input_frame": chunk.numpy(),
                    "states": state,
                    "atten_lim_db": self.atten_lim_db,
                },
            )
            enhanced_chunks.append(torch.tensor(out[0]))
            state = out[1]

        return torch.cat(enhanced_chunks)


    def denoise_tensor(self, audio: Union[np.ndarray, torch.Tensor], sr: int) -> np.ndarray:
    
        if not self.enabled or self.session is None:
            return audio if isinstance(audio, np.ndarray) else audio.cpu().numpy()

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim > 2:
            raise ValueError("Audio tensor must be 1D or 2D")

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        chunks, orig_len, _ = self._pad_and_chunk(audio)
        enhanced = self._denoise_chunks(chunks)
        d = self.fft_size - self.hop_size
        enhanced = enhanced[d : orig_len + d]

        if sr != self.target_sr:
            enhanced = torchaudio.functional.resample(enhanced, self.target_sr, 24000)

        return enhanced.cpu().numpy().astype(np.float32)

    def denoise_file(
        self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> tuple:
      
        if not self.enabled or self.session is None:
            print("[DenoiserONNX] 降噪关闭，返回原音频")
            waveform, sr = torchaudio.load(str(input_path))
            return waveform.mean(dim=0).numpy(), sr

        input_path = Path(input_path)
        audio, sr = torchaudio.load(str(input_path), channels_first=True)
        audio = audio.mean(dim=0)
        denoised = self.denoise_tensor(audio, sr)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(output_path),
                torch.tensor(denoised).unsqueeze(0),
                24000,
                encoding="PCM_S",
                bits_per_sample=16,
            )

        return denoised, sr


if __name__ == "__main__":
    import soundfile as sf

    # 1. 加载音频
    input_path = "vocals.wav"
    output_path = "denoised.wav"

    # 2. 初始化模型
    denoiser = DenoiserONNX(
        model_path="./pretrain_model/denoiser_model.onnx",
        enabled=True,
        device="cpu",  # 或 "cuda"
        target_sr=48000
    )

    # 3. 执行降噪
    enhanced, sr = denoiser.denoise_file(input_path, output_path)

    print(f"✅ 降噪完成: {output_path}, 采样率={24000}, 时长={len(enhanced)/sr:.2f}s")

