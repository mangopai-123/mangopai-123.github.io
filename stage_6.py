import torch
import librosa
import json
from datetime import datetime
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class ASR_wav2vec:
    def __init__(self, model_id="facebook/wav2vec2-xls-r-300m", device=None, log_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Wav2vecASR] Loading model '{model_id}' on {self.device} ...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id, weights_only=False)
        #self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id, weights_only=False).to(self.device)
        #self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.logs = []
        self.log_path = Path(log_path) if log_path else None

    def _transcribe(self, wav, sr=16000):
        """单段音频转录"""
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)  # 重采样至16kHz
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(inputs["input_values"], attention_mask=inputs.get("attention_mask")).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids)[0]
        return text.strip()

    def process_segments(self, vad_segments, sr):
        """
        输入:
          vad_segments: list[dict]
              [
                  {"path": "xxx.wav", "start": 1.0, "end": 3.5, "wav": np.ndarray},
                  ...
              ]
          sr: 输入音频采样率
        输出:
          更新每个 segment，添加 "asr_text" 字段，并更新日志
        """
        for seg in vad_segments:
            try:
                #continue
                text = self._transcribe(seg["wav"], sr)
                seg["asr_wav2vec"] = text
                """
                log_entry = {
                    # "time": datetime.now().isoformat(),
                    "src_file": seg.get("src_file", "unknown"),
                    "file": seg.get("path", "unknown"),
                    "start": seg["start"],
                    "end": seg["end"],
                    "asr_wav2vec": text,
                }

                if "mos_scores" in seg:
                    log_entry["mos_scores"] = seg["mos_scores"]
                if "diarization" in seg:
                    log_entry["speaker"] = seg["diarization"]

                self.logs.append(log_entry)
                """
            except Exception as e:
                print(f"[ASR] Error processing segment from {seg.get('path', 'unknown')}: {e}")
        return vad_segments, self.logs

    def save_logs(self, output_dir="logs", filename="data_clean_pipeline_results.json"):
        if not self.logs:
            print("No logs to save.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"Logs saved to {file_path}")
        return file_path


if __name__ == "__main__":

    import numpy as np

    # 模拟VAD输出
    vad_segments = [
        {"path": "data/audio_1.wav", "start": 0.0, "end": 2.0, "wav": np.random.randn(32000)},
        {"path": "data/audio_1.wav", "start": 2.0, "end": 4.5, "wav": np.random.randn(40000)},
    ]

    # 初始化并运行ASR
    asr = ASR_wav2vec(model_id="pretrain_model/Japanese/japanese-wav2vec2-base-rs35kh", log_path="logs/asr_results.json")
    updated_segments, logs = asr.process_segments(vad_segments, sr=16000)
    print(updated_segments)
