import torch
import numpy as np
import pandas as pd
from typing import Dict, Union
from pyannote.audio import Pipeline
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import warnings
from pyannote.audio.utils.reproducibility import ReproducibilityWarning
warnings.filterwarnings("ignore", category=ReproducibilityWarning)


class SpeakerDiarization:

    def __init__(
        self,
        model_config: Union[str, Dict] = "pretrain_model/pyannote_models/config.yaml",
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        print(f"[Diarization] Loading model on {self.device} ...")
        self.pipeline = Pipeline.from_pretrained(model_config).to(self.device)

    def _process_segment(self, segment_audio: np.ndarray, sr: int) -> list:
        waveform = torch.tensor(segment_audio, dtype=torch.float32).unsqueeze(0).to(self.device)

        diarization_result = self.pipeline({"waveform": waveform, "sample_rate": sr, "channel": 0})
        speaker_segments = []
        
        current_speaker = None
        current_start = None
        current_end = None

        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            start_sec, end_sec = segment.start, segment.end

            if speaker == current_speaker:
                current_end = max(current_end, end_sec)
            else:
                if current_speaker is not None:
                    speaker_segments.append({
                        "speaker": current_speaker,
                        "start_sec": current_start,
                        "end_sec": current_end
                    })
                
                current_speaker = speaker
                current_start = start_sec
                current_end = end_sec

        if current_speaker is not None:
            speaker_segments.append({
                "speaker": current_speaker,
                "start_sec": current_start,
                "end_sec": current_end
            })
        return speaker_segments


    def process(self, vad_output: Dict, sr: int) -> Dict:
        
        for seg in vad_output.get("segments", []):
            audio = seg.get("audio")
            if audio is None or len(audio) == 0:
                seg["diarization"] = []
                continue

            try:
            #if True:
                seg["src_path"] = vad_output["src_audio"]
                seg["diarization"] = self._process_segment(audio, sr)
                
            except Exception as e:
            #else:
                seg["diarization"] = []
                seg["diarization_error"] = str(e)
                print(f"[Diarization] ⚠️ Segment {seg['index']} failed: {e}")


        return vad_output


if __name__ == "__main__":
    import librosa

    from stage_3 import CoarseVADProcessor 

    wav, sr = librosa.load("vocals.wav", sr=None)
    vad = CoarseVADProcessor(local=True, auto_save=True)
    vad_result = vad.process(wav, sr, src_path="enhanced_vocal.wav")

    print(vad_result)

    diar = SpeakerDiarization(device="cuda:0")    
    diar_result = diar.process(vad_result, sr)
    print(diar_result)

