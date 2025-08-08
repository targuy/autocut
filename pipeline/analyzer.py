import sys

try:
    import cv2
except ImportError:
    print("[ERROR] Module 'cv2' non trouvé. Veuillez installer OpenCV-Python.")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("[ERROR] Module 'numpy' non trouvé. Veuillez installer NumPy.")
    sys.exit(1)
try:
    import torch
except ImportError:
    print("[ERROR] Module 'torch' non trouvé. Veuillez installer PyTorch.")
    sys.exit(1)
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Module 'ultralytics' non trouvé. Veuillez installer la librairie Ultralytics YOLO.")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline as hf_pipeline
except ImportError:
    snapshot_download = None
    AutoFeatureExtractor = None
    AutoModelForImageClassification = None
    hf_pipeline = None

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Module 'PIL' non trouvé. Veuillez installer Pillow.")
    sys.exit(1)

from detectors.mask import face_mask_percentage
from detectors.nsfw import NSFWWrapper
from detectors.pose import estimate_head_pose
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Module 'tqdm' non trouvé. Veuillez installer tqdm.")
    sys.exit(1)


def normalize_gender(label: str) -> str:
    label = label.lower()
    if "female" in label or "woman" in label:
        return "female"
    elif "male" in label or "man" in label:
        return "male"
    return label


class FrameAnalyzer:
    def __init__(
            self,
            face_bbox_weights: str,
            gender_model_id: str,
            device: str = "cuda:0",
            # Minimum visible face percentage (100 - max_mask_percentage)
            min_visible_face_threshold: float = 40.0,
            # Aire minimale de la bbox visage en % de l'image (anti sujets trop petits)
            min_face_bbox_area_pct: float = 1.0,
            gender: str = "tous",
            enable_face: bool = True,
            enable_gender: bool = True,
            enable_nsfw: bool = True,
            nsfw_mode: str = "high",
            min_gender_confidence: float = 0.8,
            min_face_confidence: float = 0.25,
            max_head_pitch: float = 20.0,
            max_head_yaw: float = 30.0,
            max_head_roll: float = 20.0,
            debug: bool = False
    ):
        self.face_th = min_visible_face_threshold
        self.min_face_bbox_area_pct = float(min_face_bbox_area_pct)
        self.gender_target = gender.lower()
        self.debug = debug
        self.enable_face = enable_face
        self.enable_gender = enable_gender
        self.enable_nsfw = enable_nsfw
        self.nsfw_mode = nsfw_mode
        self.min_gender_conf = min_gender_confidence
        self.min_face_conf = min_face_confidence
        self.max_head_pitch = max_head_pitch
        self.max_head_yaw = max_head_yaw
        self.max_head_roll = max_head_roll

        if torch.cuda.is_available() and not device.lower().startswith("cpu"):
            self.yolo_device = device if device.startswith("cuda") else f"cuda:{device}"
            if ":" in device:
                self.hf_device = int(device.split(":")[-1])
            elif device.isdigit():
                self.hf_device = int(device)
            else:
                self.hf_device = 0
        else:
            self.yolo_device = "cpu"
            self.hf_device = -1

        self.face_model = YOLO(face_bbox_weights, task="detect") if enable_face else None

        if self.enable_face:
            import importlib.util
            if importlib.util.find_spec("mediapipe") is None:
                if self.debug:
                    print("[WARN] Module mediapipe non installé - l'orientation de la tête ne sera pas évaluée.")

        if enable_nsfw:
            import importlib.util
            if importlib.util.find_spec("nsfw_image_detector") is None:
                if self.debug:
                    print("[WARN] Détection NSFW activée mais module 'nsfw_image_detector' indisponible - désactivée.")
                self.nsfw_detector = None
                self.enable_nsfw = False
            else:
                self.nsfw_detector = NSFWWrapper(device=self.yolo_device, dtype="bfloat16")
        else:
            self.nsfw_detector = None

        self.gender_clf = None
        if enable_gender:
            if snapshot_download is None or hf_pipeline is None:
                if self.debug:
                    print("[WARN] Détection de genre activée mais modules transformers absents - désactivée.")
                self.enable_gender = False
            else:
                try:
                    model_dir = snapshot_download(repo_id=gender_model_id)
                    self.gender_clf = hf_pipeline(
                        task="image-classification",
                        model=AutoModelForImageClassification.from_pretrained(model_dir),
                        feature_extractor=AutoFeatureExtractor.from_pretrained(model_dir),
                        device=self.hf_device,
                    )
                except Exception as e:
                    if self.debug:
                        print(f"[WARN] Impossible de charger le modèle de genre '{gender_model_id}' : {e}")
                    self.enable_gender = False
                    self.gender_clf = None

    def analyze_frame(self, frame: np.ndarray, t: float = 0.0) -> Dict[str, Any]:
        start_time = time.time()
        metrics: Dict[str, Any] = {
            "valid": False, "t": t,
            "mask_pct": None, "face_visible_pct": None,
            "gender": None, "gender_conf": None, "nsfw": None, "nsfw_probas": None,
            "pitch": None, "yaw": None, "roll": None,
            "proc_ms": 0.0, "reason": []
        }

        def fail(reason: str):
            metrics["valid"] = False
            metrics["reason"].append(reason)
            metrics["proc_ms"] = (time.time() - start_time) * 1000
            return metrics

        try:
            # 1) NSFW (policy finale en pipeline)
            if self.enable_nsfw and self.nsfw_detector:
                is_ns, probas = self.nsfw_detector.predict(frame, self.nsfw_mode)
                metrics["nsfw"] = is_ns
                metrics["nsfw_probas"] = probas
                if self.debug:
                    prob_dict = probas[0] if isinstance(probas, (list, tuple)) else probas
                    try:
                        prob_str = ", ".join(f"{str(k)}:{float(v):.3f}" for k, v in prob_dict.items())
                    except Exception:
                        prob_str = str(prob_dict)
                    print(f"[{t:.2f}s] NSFW probabilities ({self.nsfw_mode}): {prob_str}")
                    print(f"[{t:.2f}s] is_nsfw -> {is_ns}")

            # 2) Visage
            if self.enable_face and self.face_model is not None:
                res_face = self.face_model.predict(frame, verbose=False, device=self.yolo_device,
                                                   conf=self.min_face_conf)
                if not res_face or res_face[0].boxes.data.size(0) == 0:
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun visage détecté (conf > {self.min_face_conf})")
                    return fail("visage_absent")

                # Prendre le visage le plus grand
                boxes = res_face[0].boxes.xyxy.cpu().numpy().astype(int)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = int(np.argmax(areas))
                x1, y1, x2, y2 = boxes[idx]
                h, w = frame.shape[:2]
                face_area_pct = (areas[idx] / float(w * h)) * 100.0
                if face_area_pct < self.min_face_bbox_area_pct:
                    if self.debug:
                        print(f"[{t:.2f}s] Visage trop petit ({face_area_pct:.2f}% < seuil {self.min_face_bbox_area_pct}%)")
                    return fail("visage_trop_petit")

                face_roi_for_mask = frame[y1:y2, x1:x2]

                # Masque/visibilité
                actual_mask_pct = face_mask_percentage(face_roi_for_mask)
                metrics["mask_pct"] = actual_mask_pct
                actual_visible_pct = 100.0 - actual_mask_pct
                metrics["face_visible_pct"] = actual_visible_pct
                if actual_visible_pct < self.face_th:
                    if self.debug:
                        print(f"[{t:.2f}s] Visage trop masqué ({actual_visible_pct:.1f}% < {self.face_th}%)")
                    return fail("visage_masque")

                # Pose
                try:
                    angles = estimate_head_pose(frame, (x1, y1, x2, y2))
                    if angles:
                        metrics["pitch"], metrics["yaw"], metrics["roll"] = angles
                        if abs(metrics["pitch"]) > self.max_head_pitch or \
                                abs(metrics["yaw"]) > self.max_head_yaw or \
                                abs(metrics["roll"]) > self.max_head_roll:
                            if self.debug:
                                print(f"[{t:.2f}s] Tête hors limites (P:{metrics['pitch']:.1f}, Y:{metrics['yaw']:.1f}, R:{metrics['roll']:.1f})")
                            return fail("visage_pose")
                except Exception as e:
                    if self.debug:
                        print(f"[{t:.2f}s] Pose exception: {e}")

            # 3) Genre (sur frame; idéalement ROI visage → amélioration ultérieure)
            if self.enable_gender and self.gender_clf is not None:
                img_for_gender = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    preds = self.gender_clf(img_for_gender)
                    if not preds:
                        if self.debug:
                            print(f"[{t:.2f}s] Genre: aucune prédiction")
                        return fail("genre_no_pred")
                    pred_main = preds[0]
                    label = pred_main["label"]
                    conf = pred_main["score"]
                    metrics["gender_conf"] = conf
                    if conf < self.min_gender_conf:
                        if self.debug:
                            print(f"[{t:.2f}s] Confiance genre {conf:.2f} < {self.min_gender_conf}")
                        return fail("genre_confiance")
                    normalized_label = normalize_gender(label)
                    metrics["gender"] = normalized_label
                    if self.gender_target not in ("tous", "all") and normalized_label != self.gender_target:
                        if self.debug:
                            print(f"[{t:.2f}s] Genre {normalized_label} != filtre {self.gender_target}")
                        return fail("genre_filtre")
                except Exception as e:
                    if self.debug:
                        print(f"[{t:.2f}s] Genre exception: {e}")
                    return fail("genre_exception")

            metrics["valid"] = True
            metrics["reason"] = ["all_pass"]
            metrics["proc_ms"] = (time.time() - start_time) * 1000
            return metrics

        except Exception as e:
            if self.debug:
                print(f"[ERROR] Exception majeure analyze_frame @ {t:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return fail("exception_globale")


class VideoAnalyzer:
    def __init__(
            self,
            fa: FrameAnalyzer,
            min_dur: float,
            sample_rate: float,
            refine_rate: float,
            max_gap: float = 0.2,
            num_workers: int = 4,
    ):
        self.fa = fa
        self.min_dur = min_dur
        self.sample_rate = sample_rate
        # Nombre maximum de frames négatives consécutives, dérivé du gap temporel
        self.max_gap_frames = int(max(0.0, max_gap) * sample_rate)
        self.num_workers = num_workers

    def process(self, video_path: str, out_dir: str = "clips") -> List[Tuple[float, float]]:
        vid_path_obj = Path(video_path)
        cap = cv2.VideoCapture(str(vid_path_obj))
        if not cap.isOpened():
            print(f"[ERROR] Impossible d'ouvrir la vidéo : {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"[ERROR] FPS invalide ({fps}) pour la vidéo: {video_path}.")
            cap.release()
            return []

        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_video / fps if total_frames_video > 0 and fps > 0 else 0
        if duration == 0:
            print(f"[WARN] Durée vidéo nulle/invalide pour {video_path}.")
            cap.release()
            return []

        if self.fa.debug:
            print(f"[DEBUG] Vidéo: {video_path}, {fps:.2f} FPS, ~{total_frames_video} frames, {duration:.2f}s")

        dt_sample = 1.0 / self.sample_rate
        timestamps_to_sample = list(np.arange(0.0, duration, dt_sample))
        if not timestamps_to_sample:
            print(f"[WARN] Aucun timestamp à échantillonner pour {video_path}.")
            cap.release()
            return []

        processed_results: List[Tuple[float, Dict[str, Any]]] = []
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
                futures_map: Dict[Any, float] = {}
                pbar_submit = tqdm(total=len(timestamps_to_sample), desc=f"Soumission frames {vid_path_obj.name}",
                                   disable=not self.fa.debug)
                for t_sample in timestamps_to_sample:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t_sample * 1000)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if self.fa.debug:
                            print(f"[WARN] Lecture frame échouée @ {t_sample:.2f}s")
                        break
                    future = exe.submit(self.fa.analyze_frame, frame.copy(), t_sample)
                    futures_map[future] = t_sample
                    pbar_submit.update(1)
                pbar_submit.close()

                temp_results = []
                pbar_collect = tqdm(total=len(futures_map), desc=f"Analyse frames {vid_path_obj.name}",
                                    disable=not self.fa.debug)
                for future in futures_map:
                    ts = futures_map[future]
                    try:
                        result_metrics = future.result()
                        temp_results.append((ts, result_metrics))
                    except Exception as e:
                        print(f"[ERROR][Thread] Erreur analyse frame @ {ts:.2f}s : {e}")
                        dummy_metrics: Dict[str, Any] = {"valid": False, "t": ts, "reason": ["thread_exception"], "proc_ms": 0}
                        temp_results.append((ts, dummy_metrics))
                    pbar_collect.update(1)
                pbar_collect.close()

                processed_results = sorted(temp_results, key=lambda x: x[0])
        finally:
            cap.release()

        # Merge avec tolérance
        valid_intervals: List[Tuple[float, float]] = []
        current_segment_start_time: Optional[float] = None
        last_valid_frame_time: float = 0.0
        consecutive_invalid_frames_count: int = 0

        for t_frame, metrics in processed_results:
            if metrics["valid"]:
                if current_segment_start_time is None:
                    current_segment_start_time = t_frame
                last_valid_frame_time = t_frame
                consecutive_invalid_frames_count = 0
            else:
                if current_segment_start_time is not None:
                    consecutive_invalid_frames_count += 1
                    if consecutive_invalid_frames_count > self.max_gap_frames:
                        segment_duration = last_valid_frame_time - current_segment_start_time
                        if segment_duration >= self.min_dur:
                            valid_intervals.append((current_segment_start_time, last_valid_frame_time + dt_sample))
                        current_segment_start_time = None
                        consecutive_invalid_frames_count = 0

        if current_segment_start_time is not None:
            segment_duration = last_valid_frame_time - current_segment_start_time
            if segment_duration >= self.min_dur:
                valid_intervals.append((current_segment_start_time, last_valid_frame_time + dt_sample))

        return valid_intervals