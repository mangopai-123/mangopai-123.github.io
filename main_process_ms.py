import time
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import yaml
from tqdm import tqdm
import gc
import orjson
from itertools import chain


_GLOBALS = {}


def get_or_create_modules(config_path: str) -> Dict[str, Any]:
    if "modules" in _GLOBALS:
        return _GLOBALS["modules"]

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from stage_0 import AudioNormalizer
    from stage_1 import VocalSeparator
    from stage_2 import DenoiserONNX
    from stage_3 import CoarseVADProcessor
    from stage_4 import SpeakerDiarization
    from stage_5 import MOSPredictor
    from stage_6 import ASR_wav2vec
    from stage_7 import ASR_whisper

    pipeline_cfg = config["pipeline"]
    global_device = pipeline_cfg.get("device", "cpu")

    normalizer = AudioNormalizer(**config["modules"]["audio_normalizer"])

    vs_cfg = config["modules"]["vocal_separator"]
    separator = VocalSeparator(
        model_id=vs_cfg["model_id"],
        repo_dir=vs_cfg["repo_dir"],
        device=vs_cfg.get("device", global_device),
    )

    denoise_cfg = config["modules"]["denoiser_onnx"]
    denoiser = None
    if denoise_cfg.get("enabled", False):
        denoiser = DenoiserONNX(
            model_path=denoise_cfg["model_path"],
            enabled=True,
            device=denoise_cfg.get("device", global_device),
            target_sr=denoise_cfg["target_sr"],
            atten_lim_db=denoise_cfg["atten_lim_db"],
            hop_size=denoise_cfg["hop_size"],
            fft_size=denoise_cfg["fft_size"],
        )

    vad_cfg = config["modules"]["vad_processor"]
    vad = CoarseVADProcessor(
        save_root=vad_cfg["save_root"],
        local=vad_cfg["local"],
        model_name=vad_cfg["model_name"],
        auto_save=vad_cfg["auto_save"],
    )

    diar_cfg = config["modules"]["speaker_diarization"]
    diarizer = SpeakerDiarization(
        model_config=diar_cfg["model_config"],
        device=diar_cfg.get("device", global_device),
    )

    mos_cfg = config["modules"]["mos_predictor"]
    mos = MOSPredictor(model_path=mos_cfg["model_path"])

    asr1_cfg = config["modules"]["asr_wav2vec"]
    asr1 = ASR_wav2vec(
        model_id=asr1_cfg["model_id"],
        device=asr1_cfg.get("device", global_device),
        log_path=asr1_cfg.get("log_path"),
    )

    asr2_cfg = config["modules"]["asr_whisper"]
    asr2 = ASR_whisper(
        model_id=asr2_cfg["model_id"],
        device=asr2_cfg.get("device", global_device),
        log_path=asr2_cfg.get("log_path"),
    )

    _GLOBALS["modules"] = {
        "normalizer": normalizer,
        "separator": separator,
        "denoiser": denoiser,
        "vad": vad,
        "diarizer": diarizer,
        "mos": mos,
        "asr1": asr1,
        "asr2": asr2,
    }
    return _GLOBALS["modules"]


def process_single_file_in_worker(args):
    audio_path_str, config_path, output_dir_str, checkpoint_dir_str = args
    audio_path = Path(audio_path_str)
    output_dir = Path(output_dir_str)
    checkpoint_dir = Path(checkpoint_dir_str)
    done_file = checkpoint_dir / f"{audio_path.stem}.done"

    if done_file.exists():
        return f"{audio_path.name} 已完成，跳过"

    try:
        modules = get_or_create_modules(config_path)
        normalizer = modules["normalizer"]
        separator = modules["separator"]
        denoiser = modules["denoiser"]
        vad = modules["vad"]
        diarizer = modules["diarizer"]
        mos = modules["mos"]
        asr1 = modules["asr1"]
        asr2 = modules["asr2"]

        result = normalizer.standardization(str(audio_path))
        if not result:
            raise RuntimeError("标准化失败")

        res_vad = vad.process_first(result["waveform"], result["sample_rate"])

        for item in res_vad["segments"]:
            wav_part =  item["audio"]
            vocals, sr = separator.source_separation(wav_part, result["sample_rate"])
            if vocals is None:
                print("⚠️ 人声分离 Error")
                continue

            if denoiser is not None:
                vocals = denoiser.denoise_tensor(vocals, sr)

            vad_result = vad.process(vocals, sr, str(audio_path))
            if not vad_result["segments"]:
                print("⚠️ 无语音片段 Error")
                continue
        
            diar_result = diarizer.process(vad_result, result["sample_rate"])

            segments = []
            for seg in diar_result["segments"]:
                segments.append({
                    "src_file": seg["src_path"],
                    "path": seg["path"],
                    "start": seg["start_sec"],
                    "end": seg["end_sec"],
                    "diarization": seg["diarization"],
                    "wav": seg["audio"],
                })

            segments, _ = mos.predict_segments(segments, result["sample_rate"])
            segments, _ = asr1.process_segments(segments, result["sample_rate"])
            segments, _ = asr2.process_segments(segments, result["sample_rate"])

        asr2.save_logs(output_dir=str(output_dir), filename=f"{audio_path.stem}_asr.json")
        done_file.touch()

        return f"{audio_path.name} ✅ 处理完成"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"{audio_path.name} ❌ 处理失败: {e}"


class AudioProcessingPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        cfg = self.config["pipeline"]
        self.input_dir = Path(cfg["input_dir"])
        self.output_dir = Path(cfg["output_dir"])
        self.checkpoint_dir = Path(cfg["checkpoint_dir"])
        self.num_workers = cfg.get("num_workers", 1)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INIT] 多进程音频 Pipeline 启动，workers={self.num_workers}")

 
    def run(self):
        # audio_files = sorted(self.input_dir.glob("*.wav"))
        audio_files = sorted(chain(self.input_dir.glob("*.wav"), self.input_dir.glob("*.mp3")))
        total = len(audio_files)
        if total == 0:
            print("❗ 未找到 .wav 文件")
            return

        num_workers = min(self.num_workers, multiprocessing.cpu_count())
        max_pending = max(2 * num_workers, 1)
        retry_limit = 1

        print(f"🎧 待处理文件数: {total:,}")
        print(f"🚀 启动 {num_workers} 个进程，每次最多挂起 {max_pending} 个任务\n")

        def task_gen():
            for audio_path in audio_files:
                yield (str(audio_path), self.config_path, str(self.output_dir), str(self.checkpoint_dir))

        tasks = task_gen()
        errors = []
        finished = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pending = set()
            pbar = tqdm(total=total, desc="Pipeline 进度", ncols=100)

            # 预加载任务
            for _ in range(min(max_pending, total)):
                try:
                    args = next(tasks)
                    fut = executor.submit(process_single_file_in_worker, args)
                    fut._args = args
                    fut._retries = 0
                    pending.add(fut)
                except StopIteration:
                    break

            while pending:
                for fut in as_completed(pending):
                    pending.remove(fut)
                    try:
                        msg = fut.result(timeout=30)
                        # print(msg)
                    except Exception as e:
                        args = getattr(fut, "_args", None)
                        retries = getattr(fut, "_retries", 0)
                        msg = f"[Error] {args[0] if args else 'unknown'}: {e}"
                        print(msg)
                        errors.append(msg)

                        if retries < retry_limit:
                            new_fut = executor.submit(process_single_file_in_worker, args)
                            new_fut._args = args
                            new_fut._retries = retries + 1
                            pending.add(new_fut)

                    finished += 1
                    pbar.update(1)

                    try:
                        args = next(tasks)
                        new_fut = executor.submit(process_single_file_in_worker, args)
                        new_fut._args = args
                        new_fut._retries = 0
                        pending.add(new_fut)
                    except StopIteration:
                        pass

                    if finished % 200 == 0:
                        gc.collect()

            pbar.close()

        if errors:
            log_path = self.output_dir / "pipeline_errors.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(errors) + "\n")
            print(f"⚠️ 共 {len(errors)} 个任务失败，错误日志已保存到 {log_path}")

        print("\n🧩 所有任务完成，开始合并 ASR 结果...")
        self._merge_asr_results()

   
    def _merge_asr_results(self, chunk_limit: int = 10000):
        asr_files = sorted(self.output_dir.glob("*_asr.json"))
        total = len(asr_files)
        print(f"检测到 {total:,} 个 ASR 文件，将分块合并（每块 {chunk_limit:,} 个）")

        if total == 0:
            print("⚠️ 无 ASR 文件可合并")
            return

        for i in range(0, len(asr_files), chunk_limit):
            chunk = asr_files[i : i + chunk_limit]
            idx = i // chunk_limit + 1
            out_path = self.output_dir / f"data_part_{idx:03d}.json"
            tmp_path = out_path.with_suffix(".tmp")

            if out_path.exists():
                print(f"⏭️  {out_path.name} 已存在，跳过")
                continue

            print(f"📦 处理分块 {idx} ({len(chunk)} 文件)...")
            with open(tmp_path, "wb") as fout:
                fout.write(b"{\n")
                written = 0
                with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
                    futures = {pool.submit(self._load_json_fast, p): p for p in chunk}
                    for fut in tqdm(as_completed(futures), total=len(chunk), desc=f"Chunk {idx}", unit="file", leave=False):
                        name, data = fut.result()
                        if name and data:
                            fout.write(f'  "{name}": '.encode() + orjson.dumps(data) + b",\n")
                            written += 1
                if written:
                    fout.seek(fout.tell() - 2)
                fout.write(b"\n}\n")
            tmp_path.replace(out_path)
            print(f"✅ 分块 {idx} 完成 ({written} 条)")

        print("🎯 所有分块完成。正在合并为总文件...")
        final = self.output_dir / "data_clean_pipeline_results.json"
        tmp_final = final.with_suffix(".tmp")
        with open(tmp_final, "wb") as fout:
            fout.write(b"{\n")
            first = True
            for part in sorted(self.output_dir.glob("data_part_*.json")):
                with open(part, "rb") as fin:
                    c = fin.read().strip()
                    if c.startswith(b"{"):
                        c = c[1:]
                    if c.endswith(b"}"):
                        c = c[:-1]
                    if not c:
                        continue
                    if not first:
                        fout.write(b",\n")
                    else:
                        first = False
                    fout.write(c)
            fout.write(b"\n}\n")
        tmp_final.replace(final)
        print(f"🎉 合并完成: {final}")

    def _load_json_fast(self, path: Path):
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
            return path.stem.replace("_asr", ""), data
        except Exception:
            return None, None


if __name__ == "__main__":
    ss = time.time()
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="多进程音频处理 Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    AudioProcessingPipeline(args.config).run()

    print(time.time() - ss)
