# pipeline_orchestrator.py
import os
import json
import argparse
from pathlib import Path
import numpy as np
import time
from typing import Dict, Any, Optional

import yaml  # 需要安装：pip install pyyaml


from stage_0 import AudioNormalizer
from stage_1 import VocalSeparator
from stage_2 import DenoiserONNX
from stage_3 import CoarseVADProcessor
from stage_4 import SpeakerDiarization
from stage_5 import MOSPredictor
from stage_6 import ASR_wav2vec
from stage_7 import ASR_whisper

class AudioProcessingPipeline:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        pipeline_cfg = self.config["pipeline"]
        self.input_dir = Path(pipeline_cfg["input_dir"])
        self.output_dir = Path(pipeline_cfg["output_dir"])
        self.checkpoint_dir = Path(pipeline_cfg["checkpoint_dir"])
        self.resume = pipeline_cfg["resume"]
        self.global_device = pipeline_cfg.get("device", "cpu")
        self.num_workers = pipeline_cfg.get("num_workers", 1)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        norm_cfg = self.config["modules"]["audio_normalizer"]
        self.normalizer = AudioNormalizer(**norm_cfg)

        vs_cfg = self.config["modules"]["vocal_separator"]
        vs_device = vs_cfg.get("device", self.global_device)
        self.separator = VocalSeparator(
            model_id=vs_cfg["model_id"],
            repo_dir=vs_cfg["repo_dir"],
            device=vs_device
        )

        denoise_cfg = self.config["modules"]["denoiser_onnx"]
        denoise_device = denoise_cfg.get("device", self.global_device)
        self.denoiser = DenoiserONNX(
            model_path=denoise_cfg["model_path"],
            enabled=denoise_cfg["enabled"],
            device=denoise_device,
            target_sr=denoise_cfg["target_sr"],
            atten_lim_db=denoise_cfg["atten_lim_db"],
            hop_size=denoise_cfg["hop_size"],
            fft_size=denoise_cfg["fft_size"]
        )

        vad_cfg = self.config["modules"]["vad_processor"]
        self.vad = CoarseVADProcessor(
            save_root=vad_cfg["save_root"],
            local=vad_cfg["local"],
            model_name=vad_cfg["model_name"],
            auto_save=vad_cfg["auto_save"]
        )

        diar_cfg = self.config["modules"]["speaker_diarization"]
        diar_device = diar_cfg.get("device", self.global_device)
        self.diarizer = SpeakerDiarization(
            model_config=diar_cfg["model_config"],
            device=diar_device
        )

        mos_cfg = self.config["modules"]["mos_predictor"]
        self.mos = MOSPredictor(model_path=mos_cfg["model_path"])

        asr_cfg = self.config["modules"]["asr_wav2vec"]
        asr_device = asr_cfg.get("device", self.global_device)
        self.asr1 = ASR_wav2vec(
            model_id=asr_cfg["model_id"],
            device=asr_device,
            log_path=asr_cfg.get("log_path")
        )

        asr_cfg = self.config["modules"]["asr_whisper"]
        asr_device = asr_cfg.get("device", self.global_device)
        self.asr2 = ASR_whisper(
            model_id=asr_cfg["model_id"],
            device=asr_device,
            log_path=asr_cfg.get("log_path")
        )

        print("AudioProcessingPipeline Init Successfull...")
        time.sleep(1)

    def _is_completed(self, audio_path: Path) -> bool:
        done_file = self.checkpoint_dir / f"{audio_path.stem}.done"
        return done_file.exists()

    def _mark_as_completed(self, audio_path: Path):
        done_file = self.checkpoint_dir / f"{audio_path.stem}.done"
        done_file.touch()

    def _extract_audio_segments(self, vad_result: Dict) -> list:
        segments = []
        for seg in vad_result["segments"]:
            segments.append({
                "src_file": seg["src_path"],
                "path": seg["path"],
                "start": seg["start_sec"],
                "end": seg["end_sec"],
                "diarization": seg['diarization'],
                "wav": seg["audio"],
            })
        return segments
    
    def process_single_file(self, audio_path: Path):

        if self._is_completed(audio_path):
            print(f"✅ {audio_path.name} 已完成，跳过")
            return

        print(f"\n🔄 处理文件: {audio_path.name}")

        try:
            result = self.normalizer.standardization(str(audio_path))
            if not result:
                raise RuntimeError("标准化失败")
            
            vocals, sr = self.separator.source_separation(
                result["waveform"], result["sample_rate"]
            )
            if vocals is None:
                raise RuntimeError("人声分离失败")

            denoised = vocals
        
            # denoised = self.denoiser.denoise_tensor(vocals, sr)

            vad_result = self.vad.process(denoised, sr, str(audio_path))
            if not vad_result["segments"]:
                raise RuntimeError("无语音片段")

            diar_result = self.diarizer.process(vad_result, result["sample_rate"])

            segments = self._extract_audio_segments(diar_result)

            segments, _ = self.mos.predict_segments(segments, result["sample_rate"])

            segments, _ = self.asr1.process_segments(segments, result["sample_rate"])
            
            segments, _ = self.asr2.process_segments(segments, result["sample_rate"])
        
            self._mark_as_completed(audio_path)
        except Exception as e:
            print(f"处理 {audio_path.name} 失败: {e}")

    def run(self):
        audio_files = list(self.input_dir.glob("*.wav"))
        if not audio_files:
            print(" .wav 文件")
            return

        print(f"Total: {len(audio_files)} files")
        for audio_path in sorted(audio_files):
            try:
                self.process_single_file(audio_path)
            except Exception as e:
                print(f"{audio_path} ERROR: {e}")
                continue

        self.asr2.save_logs(output_dir=str(self.output_dir), filename="data_clean_pipeline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于配置文件的音频处理 Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    pipeline = AudioProcessingPipeline(config_path=args.config)
    pipeline.run()
