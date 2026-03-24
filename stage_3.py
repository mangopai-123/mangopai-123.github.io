import torch
import numpy as np
import torchaudio
import librosa
import uuid
import datetime
from scipy.io.wavfile import write
from pathlib import Path
from typing import List, Dict, Union, Optional


class CoarseVADProcessor:
    def __init__(
        self,
        save_root: Union[str, Path] = "./vad_output",
        local: bool = False,
        model_name: str = "silero_vad",
        auto_save: bool = False,
    ):
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        try:
            repo_or_dir = "snakers4/silero-vad" if not local else "pretrain_model/vad/silero-vad/"
            self.vad_model, utils = torch.hub.load(
                repo_or_dir=repo_or_dir,
                model=model_name,
                force_reload=False,
                onnx=True,
                source="github" if not local else "local",
            )
            (get_speech_timestamps, _, _, _, _) = utils
            self.get_speech_timestamps = get_speech_timestamps
            print(f"[VAD] Silero VAD model loaded ({'local' if local else 'remote'})")
        except Exception as e:
            raise RuntimeError(f"[VAD] Failed to load Silero VAD model: {e}")

    def _run_vad(self, audio_16k: np.ndarray) -> List[Dict[str, float]]:
        return self.get_speech_timestamps(
            audio_16k,
            self.vad_model,
            sampling_rate=16000,
            threshold=0.3,
            min_speech_duration_ms=300,
            min_silence_duration_ms=200,
            speech_pad_ms=25,
            max_speech_duration_s=30,
            return_seconds=True,
        )


    def _run_vad_first(self, audio_16k: np.ndarray) -> List[Dict[str, float]]:
        return self.get_speech_timestamps(
            audio_16k,
            self.vad_model,
            sampling_rate=16000,
            threshold=0.3,
            min_speech_duration_ms=400,
            min_silence_duration_ms=400,
            speech_pad_ms=50,
            max_speech_duration_s=100,
            return_seconds=True,
        )
    
    def _run_vad_fine(self, audio_16k: np.ndarray) -> List[Dict[str, float]]:
        return self.get_speech_timestamps(
            audio_16k,
            self.vad_model,
            sampling_rate=16000,
            threshold=0.3,
            min_speech_duration_ms=200,
            min_silence_duration_ms=100,
            speech_pad_ms=5,
            max_speech_duration_s=30,
            return_seconds=True,
        )

    def merge_vad_segments(self, segments, min_gap=0.2, min_duration=1.0, max_duration=30.0):
        if not segments:
            return []

        segments = sorted(segments, key=lambda x: x['start'])

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg['start'] - prev['end']
            if gap < min_gap and seg['end'] - prev['start'] <= max_duration:
                prev['end'] = seg['end']
            else:
                merged.append(seg)

        result = []
        i = 0
        while i < len(merged):
            seg = merged[i]
            duration = seg['end'] - seg['start']

            if duration < min_duration:
                if i + 1 < len(merged):
                    merged[i + 1]['start'] = min(merged[i + 1]['start'], seg['start'])
                elif result:
                    result[-1]['end'] = max(result[-1]['end'], seg['end'])
                else:
                    result.append(seg)
            else:
                if duration <= max_duration:
                    result.append(seg)
                # else:
                #     print(seg)
                    # result.append(seg)
            i += 1

        return result

    def refine_long_segments_16k(self, audio_16k: np.ndarray, segments: List[Dict], max_duration: float = 30.0) -> List[Dict]:
        refined_segments = []
        for seg in segments:
            start_sec, end_sec = seg["start"], seg["end"]
            duration = end_sec - start_sec

            if duration <= max_duration:
                refined_segments.append(seg)
            else:
                start_sample = int(start_sec * 16000)
                end_sample = int(end_sec * 16000)
                clip_16k = audio_16k[start_sample:end_sample]

                fine_timestamps = self._run_vad_fine(clip_16k)

                for ft in fine_timestamps:
                    refined_segments.append({
                        "start": start_sec + ft["start"],
                        "end": start_sec + ft["end"]
                    })

        return self.merge_vad_segments(refined_segments)

    def process(
        self,
        wav: Union[np.ndarray, torch.Tensor],
        sr: int,
        src_path: Union[str, Path],
        session_id: Optional[str] = None,
    ) -> Dict:

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        
        max_val = np.max(np.abs(wav))
        if max_val > 1.0:
            wav = wav / max_val

        audio_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        timestamps = self._run_vad(audio_16k)
        timestamps = self.refine_long_segments_16k(audio_16k, timestamps)

        date_str = datetime.date.today().strftime("%Y%m%d")
        session_id = session_id or uuid.uuid4().hex[:6]
        
        if not timestamps:
            return {
                "session_id": session_id,
                "src_audio": str(src_path),
                "segments": [],
            }

        segments = []
        
        for i, seg in enumerate(timestamps):
            start_sec, end_sec = seg["start"], seg["end"]
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            clip = wav[start_sample:end_sample]

            save_path = None
            if self.auto_save:
                filename = f"{date_str}_{session_id}_{i:03d}.wav"
                save_path = self.save_root / filename
                clip = clip.astype(np.float32)
                peak = np.max(np.abs(clip))
                if peak > 0:
                    target_peak = 10 ** (-3.0 / 20)
                    clip *= target_peak / peak
                    clip_ = (clip * 32767).astype(np.int16)
                write(str(save_path), sr, clip_)

            segments.append({
                "index": i,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "audio": clip,
                "path": str(save_path) if save_path else None,
            })

        return {
            "session_id": session_id,
            "src_audio": str(src_path),
            "segments": segments,
        }


    def process_first(
        self,
        wav: Union[np.ndarray, torch.Tensor],
        sr: int,
    ) -> Dict:

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        
        max_val = np.max(np.abs(wav))
        if max_val > 1.0:
            wav = wav / max_val

        audio_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        timestamps = self._run_vad_first(audio_16k)
        
        if not timestamps:
           return {
                "segments": [{
                    "audio": wav.astype(np.float32),
                    "start": 0.0,
                    "end": len(wav) / sr,
                }]
            }

        segments = []
        for i, seg in enumerate(timestamps):
            start_sec, end_sec = seg["start"], seg["end"]
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            clip = wav[start_sample:end_sample]

           
            clip = clip.astype(np.float32)
            peak = np.max(np.abs(clip))
            if peak > 0:
                target_peak = 10 ** (-3.0 / 20)
                clip *= target_peak / peak

            segments.append({
                "audio": clip,
                "start": start_sec,
                "end": end_sec,
                "duration": end_sec - start_sec,
            })

        return {
            "segments": segments,
        }

if __name__ == "__main__":
    import librosa, json

    vad = CoarseVADProcessor(local=True, auto_save=True)
    # wav, sr = librosa.load("input/audio_000019.wav", sr=None)
    wav, sr = librosa.load("/mnt/dataset/tts_data/THAI2_MP3/-Ns37dQOqg-dBDxs.mp3", sr=None)

    result = vad.process(wav, sr, src_path="vad.wav")
    
    for seg in result["segments"]:
        seg.pop("audio")

    print(json.dumps(result, indent=2, ensure_ascii=False))
