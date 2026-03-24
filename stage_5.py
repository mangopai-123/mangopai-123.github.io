import librosa
import numpy as np
import onnxruntime as ort
from datetime import datetime

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path=None) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.use_personalized = False

    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, fs, is_personalized_MOS=False):
        SAMPLING_RATE = 16000
        INPUT_LENGTH = 9.01
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg, predicted_mos_bak_seg, predicted_mos_ovr_seg = [], [], []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples):int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            oi = {'input_1': input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)

            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {
            "OVRL": float(np.mean(predicted_mos_ovr_seg)),
            "SIG": float(np.mean(predicted_mos_sig_seg)),
            "BAK": float(np.mean(predicted_mos_bak_seg))
        }
        return clip_dict


class MOSPredictor:
    def __init__(self, model_path="pretrain_model/DNSMOS/DNSMOS/sig_bak_ovr.onnx"):
        self.models = {}
        self.models["dnsmos"] = ComputeScore(model_path)
        self.logs = []

    def predict_segments(self, vad_segments, sr):
        """
        输入：
          vad_segments: list[dict] = [
            {"start": 0.5, "end": 2.3, "wav": np.ndarray, "path": 原始文件路径},
            ...
          ]
        输出：
          更新每个 segment 的 MOS 打分并记录日志（内存）
        """
        for seg in vad_segments:
            seg_wav = seg["wav"]
            if sr != 16000:
                seg_wav_16k = librosa.resample(seg_wav, orig_sr=sr, target_sr=16000)
            else:
                seg_wav_16k = seg_wav

            dnsmos_scores = self.models["dnsmos"](seg_wav_16k, 16000, False)
            seg["mos_scores"] = dnsmos_scores
            
            log_entry = {
                "time": datetime.now().isoformat(),
                "file": seg.get("path", "unknown"),
                "segment_start": seg["start"],
                "segment_end": seg["end"],
                "scores": dnsmos_scores,
            }
            self.logs.append(log_entry)

        return vad_segments, self.logs



if __name__ == "__main__":

    import numpy as np
    vad_segments = [
        {"start": 0.0, "end": 2.0, "wav": np.random.randn(32000), "path": "data/audio_1.wav"},
        {"start": 2.0, "end": 4.5, "wav": np.random.randn(40000), "path": "data/audio_1.wav"},
    ]

    mos_predictor = MOSPredictor()
    updated_segments, logs = mos_predictor.predict_segments(vad_segments, sr=16000)

    for log in logs:
        print(log)

