import torch
import librosa
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class ASR_whisper:
    def __init__(
        self,
        model_id="Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small",
        device=None,
        log_path=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[WhisperASR] Loading model '{model_id}' on {self.device} ...")

        self.cache_dir = "pretrain_model/Arabic"

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir, weights_only=False)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=self.cache_dir, weights_only=False).to(self.device)
        #self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
        #self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=self.cache_dir).to(self.device)
        self.model.eval()

        self.logs = []
        self.log_path = Path(log_path) if log_path else None

    def _transcribe(self, wav, sr=16000):
        """单段音频转录"""
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        # 准备输入
        inputs = self.processor(
            wav,
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.device)

        # 生成文本
        with torch.no_grad():
            #  if self.device.startswith("cuda"):
            #     torch.cuda.synchronize()
            generated_ids = self.model.generate(**inputs, do_sample=False)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text.strip()

    def process_segments(self, vad_segments, sr):
        """
        批量处理语音片段
        输入:
            vad_segments: list[dict]
                [
                    {"path": "xxx.wav", "start": 1.0, "end": 3.5, "wav": np.ndarray},
                    ...
                ]
            sr: 输入音频采样率
        输出:
            更新每个 segment，添加 "asr_text" 字段，并记录日志
        """
        for seg in vad_segments:
            try:
                text = self._transcribe(seg["wav"], sr)
                seg["asr_whisper"] = text

                log_entry = {
                    "src_file": seg.get("src_file", "unknown"),
                    "file": seg.get("path", "unknown"),
                    "start": seg["start"],
                    "end": seg["end"],
                    "asr_wav2vec": seg.get("asr_wav2vec", "unknown"),
                    "asr_whisper": seg.get("asr_whisper", "unknown"),
                }

                if "mos_scores" in seg:
                    log_entry["mos_scores"] = seg["mos_scores"]
                if "diarization" in seg:
                    log_entry["speaker"] = seg["diarization"]

                self.logs.append(log_entry)
            except Exception as e:
                print(f"[WhisperASR] Error processing segment from {seg.get('path', 'unknown')}: {e}")

        return vad_segments, self.logs

    def save_logs(self, output_dir="logs", filename="whisper_asr_results.json"):
        if not self.logs:
            print("No logs to save.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

        # print(f"Logs saved to {file_path}")
        self.logs = []
        return file_path


if __name__ == "__main__":
    # 模拟VAD输出测试
    vad_segments = [
        {"path": "data/audio_1.wav", "start": 0.0, "end": 4.0, "wav": np.random.randn(76000)},
        {"path": "data/audio_1.wav", "start": 2.0, "end": 4.5, "wav": np.random.randn(40000)},
    ]

    asr = ASR_whisper(
        model_id="pretrain_model/Japanese/whisper-small-japanese/",
        log_path="logs/whisper_asr_results.json"
    )

    updated_segments, logs = asr.process_segments(vad_segments, sr=16000)

    for entry in logs:
        print(entry)

    asr.save_logs()

