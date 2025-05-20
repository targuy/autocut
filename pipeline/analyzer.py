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
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any  # Added Dict, Any for metrics
import time

# Gérer l'absence éventuelle de YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Module 'ultralytics' non trouvé. Veuillez installer la librairie Ultralytics YOLO.")
    sys.exit(1)

# Gérer l'absence éventuelle des modules HuggingFace (genre)
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

# Assuming these imports are from your project structure
# Ensure these paths are correct or adjust as necessary if they are in subdirectories
from segmenters.skin import SkinSegmenter
from detectors.mask import face_mask_percentage
from detectors.nsfw import NSFWWrapper
from detectors.pose import estimate_head_pose  # This function expects (image, face_box_tuple)
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Module 'tqdm' non trouvé. Veuillez installer tqdm.")
    sys.exit(1)


def normalize_gender(label: str) -> str:
    """Normalise les libellés de genre retournés par le modèle (male/female)."""
    label = label.lower()
    if "female" in label or "woman" in label:
        return "female"
    elif "male" in label or "man" in label:
        return "male"
    return label


class FrameAnalyzer:
    def __init__(
            self,
            person_segm_weights: str,
            face_bbox_weights: str,
            skin_segm_weights: str,
            gender_model_id: str,
            device: str = "cuda:0",
            # Renamed for clarity based on Phase 2 discussion:
            # this is for person area coverage in the frame
            body_coverage_threshold: float = 10.0,  # New name reflecting purpose
            # this is for skin percentage on the detected person
            min_skin_pct_threshold: float = 50.0,  # New parameter for specific skin threshold
            # this is minimum visible face percentage (100 - max_mask_percentage)
            min_visible_face_threshold: float = 40.0,
            gender: str = "tous",
            enable_body: bool = True,
            enable_skin: bool = True,
            enable_face: bool = True,
            enable_gender: bool = True,
            enable_nsfw: bool = True,
            nsfw_mode: str = "high",
            min_gender_confidence: float = 0.8,
            min_face_confidence: float = 0.25,  # This will be used for YOLO's 'conf' parameter
            max_head_pitch: float = 20.0,
            max_head_yaw: float = 30.0,
            max_head_roll: float = 20.0,
            debug: bool = False
    ):
        # Seuils et flags
        self.body_th = body_coverage_threshold  # Threshold for person area coverage in frame
        self.skin_pct_th = min_skin_pct_threshold / 100.0  # Convert to 0.0-1.0 scale if skin_pct is 0-1
        # Or keep as 0-100 if skin_pct is 0-100.
        # Assuming SkinSegmenter.detect returns skin_pct as 0.0-100.0
        self.face_th = min_visible_face_threshold  # Minimum percentage of VISIBLE face required
        self.gender_target = gender.lower()
        self.debug = debug
        self.enable_body = enable_body
        self.enable_skin = enable_skin
        self.enable_face = enable_face
        self.enable_gender = enable_gender
        self.enable_nsfw = enable_nsfw
        self.nsfw_mode = nsfw_mode
        self.min_gender_conf = min_gender_confidence  #
        self.min_face_conf = min_face_confidence  # For YOLO face detection confidence
        self.max_head_pitch = max_head_pitch
        self.max_head_yaw = max_head_yaw
        self.max_head_roll = max_head_roll

        # Choix du device (CPU/GPU)
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

        # Chargement des modèles YOLO/segmentation
        self.person_model = YOLO(person_segm_weights, task="segment") if enable_body else None
        self.face_model = YOLO(face_bbox_weights, task="detect") if enable_face else None
        self.skin_model = SkinSegmenter(skin_segm_weights, device=self.yolo_device) if enable_skin else None

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
                self.enable_nsfw = False  # Disable if module not found
            else:
                self.nsfw_detector = NSFWWrapper(device=self.yolo_device, dtype="bfloat16")
        else:
            self.nsfw_detector = None

        self.gender_clf = None
        if enable_gender:
            if snapshot_download is None or hf_pipeline is None:
                if self.debug:
                    print("[WARN] Détection de genre activée mais modules transformers absents - désactivée.")
                self.enable_gender = False  # Disable if modules not found
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
                    self.enable_gender = False  # Disable on loading error
                    self.gender_clf = None

    def analyze_frame(self, frame: np.ndarray, t: float = 0.0) -> Dict[str, Any]:
        """Analyse une frame vidéo à l'instant t et retourne des métriques et un drapeau valid/invalid."""
        start_time = time.time()  # Renamed to avoid conflict with 't' argument
        metrics: Dict[str, Any] = {
            "valid": False, "t": t,
            "skin_pct": None, "mask_pct": None, "face_visible_pct": None,
            "gender": None, "gender_conf": None, "nsfw": None, "nsfw_probas": None,
            "pitch": None, "yaw": None, "roll": None,
            "proc_ms": 0.0, "reason": []
        }
        face_bbox_for_pose: Optional[Tuple[int, int, int, int]] = None

        def fail(reason: str):
            metrics["valid"] = False
            metrics["reason"].append(reason)
            metrics["proc_ms"] = (time.time() - start_time) * 1000
            return metrics

        try:
            # 1) NSFW detection (drop SFW frames)
            if self.enable_nsfw and self.nsfw_detector:
                is_ns, probas = self.nsfw_detector.predict(frame, self.nsfw_mode)
                metrics["nsfw"] = is_ns
                metrics["nsfw_probas"] = probas
                if self.debug:
                    prob_dict = probas[0] if isinstance(probas, (list, tuple)) else probas
                    prob_str = ", ".join(
                        f"{lvl.name}:{score:.3f}" for lvl, score in prob_dict.items())  # Assuming prob_dict has .name
                    print(f"[{t:.2f}s] NSFW probabilities ({self.nsfw_mode}): {prob_str}")
                    print(f"[{t:.2f}s] is_nsfw -> {is_ns}")
                if not is_ns:
                    return fail("nsfw")

            # 2) Détection du corps (segmentation personnes)
            if self.enable_body and self.person_model is not None:
                res_person = self.person_model.predict(frame, verbose=False, device=self.yolo_device)
                if not res_person or res_person[0].masks is None or res_person[0].masks.data.size(0) == 0:
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun corps détecté")
                    return fail("personne")

                person_mask = res_person[0].masks.data[0].cpu().numpy()  # Assuming one primary person
                body_area_pct = (person_mask.sum() / person_mask.size) * 100
                if body_area_pct < self.body_th:
                    if self.debug:
                        # Corrected log message
                        print(f"[{t:.2f}s] Zone personne {body_area_pct:.1f}% < seuil {self.body_th}%")
                    return fail("corps_taille")  # Clarified reason

            # 3) Détection visage (bbox)
            detected_face = False
            if self.enable_face and self.face_model is not None:
                # Apply min_face_conf here
                res_face = self.face_model.predict(frame, verbose=False, device=self.yolo_device,
                                                   conf=self.min_face_conf)
                if not res_face or res_face[0].boxes.data.size(0) == 0:
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun visage détecté (avec conf > {self.min_face_conf})")
                    return fail("visage_absent")

                detected_face = True
                # Assuming primary face for mask and pose
                x1, y1, x2, y2 = res_face[0].boxes.xyxy[0].cpu().numpy().astype(int)
                face_roi_for_mask = frame[y1:y2, x1:x2]
                face_bbox_for_pose = (x1, y1, x2, y2)  # Store for pose estimation

                # Check face mask percentage
                # face_mask_percentage returns percentage of MASKED pixels (0-100)
                actual_mask_pct = face_mask_percentage(face_roi_for_mask)
                metrics["mask_pct"] = actual_mask_pct
                actual_visible_pct = 100.0 - actual_mask_pct
                metrics["face_visible_pct"] = actual_visible_pct

                # self.face_th is the minimum VISIBLE percentage required
                if actual_visible_pct < self.face_th:
                    if self.debug:
                        print(
                            f"[{t:.2f}s] Visage trop masqué ({actual_visible_pct:.1f}% visible < seuil {self.face_th}%)")
                    return fail("visage_masque")

            # 4) Détection surface de peau (only if skin detection is enabled)
            if self.enable_skin and self.skin_model is not None:
                # skin_model.detect returns (skin_mask, skin_percentage_of_total_image)
                # Assuming skin_percentage is 0-100
                skin_mask_img, skin_area_pct = self.skin_model.detect(frame)
                metrics["skin_pct"] = skin_area_pct  # This is % of total image by default by SkinSegmenter
                # If you want skin % on person, it needs person mask

                # Using self.skin_pct_th (from min_person_skin_percentage config)
                # Note: This self.skin_pct_th assumes skin_area_pct is also 0-100.
                # And this is skin percentage of *whole image*. If you want skin percentage *on person area*,
                # then skin_model.detect would need to be refined or an additional calculation here.
                # For now, comparing skin_area_pct of WHOLE IMAGE with self.skin_pct_th
                if skin_area_pct < self.skin_pct_th:  #
                    if self.debug:
                        print(
                            f"[{t:.2f}s] Peau visible (sur image) {skin_area_pct:.1f}% < seuil {self.skin_pct_th}%")  #
                    return fail("peau_insuffisante")

            # 5) Orientation de la tête (si visage détecté et module activé)
            if self.enable_face and detected_face and face_bbox_for_pose is not None:  #
                try:
                    # Call with face_bbox_for_pose
                    angles = estimate_head_pose(frame, face_bbox_for_pose)
                    if angles:  # estimate_head_pose might return None
                        metrics["pitch"], metrics["yaw"], metrics["roll"] = angles
                        if abs(metrics["pitch"]) > self.max_head_pitch or \
                                abs(metrics["yaw"]) > self.max_head_yaw or \
                                abs(metrics["roll"]) > self.max_head_roll:
                            if self.debug:
                                print(
                                    f"[{t:.2f}s] Tête hors limites (P:{metrics['pitch']:.1f}, Y:{metrics['yaw']:.1f}, R:{metrics['roll']:.1f})")
                            return fail("visage_pose")
                    else:
                        if self.debug:
                            print(f"[{t:.2f}s] Échec de l'estimation de la pose de la tête.")
                        # Optionally fail here, or allow if pose estimation isn't critical: return fail("visage_pose_estimation_echec")
                except Exception as e:
                    if self.debug:
                        print(f"[{t:.2f}s] Exception lors de l'estimation de la pose: {e}")
                    # Optionally fail: return fail("visage_pose_exception")
                    pass  # Silent failure if mediapipe is missing or fails internally

            # 6) Classification de genre (si demandé et visage détecté)
            # Note: Gender classification usually performs best on clear face ROIs.
            # Consider passing face_roi if available and gender_clf supports it,
            # or ensure gender_clf can handle full frames if face detection is off.
            if self.enable_gender and self.gender_clf is not None:
                # Using full frame for now. For better accuracy, use face_roi if available.
                # face_roi_for_gender = frame[y1:y2, x1:x2] if detected_face else frame
                img_for_gender = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                try:
                    preds = self.gender_clf(img_for_gender)  # Might return a list of dicts
                    if not preds:
                        if self.debug: print(f"[{t:.2f}s] Modèle de genre n'a rien retourné.")
                        return fail("genre_no_pred")

                    pred_main = preds[0]  # Take the first prediction
                    label = pred_main["label"]
                    conf = pred_main["score"]
                    metrics["gender_conf"] = conf

                    # Using self.min_gender_conf
                    if conf < self.min_gender_conf:
                        if self.debug:
                            print(f"[{t:.2f}s] Confiance genre {conf:.2f} < seuil {self.min_gender_conf}")
                        return fail("genre_confiance")

                    normalized_label = normalize_gender(label)
                    metrics["gender"] = normalized_label
                    if self.gender_target not in ("tous", "all"):
                        if normalized_label != self.gender_target:
                            if self.debug:
                                print(f"[{t:.2f}s] Genre détecté {normalized_label} != filtre {self.gender_target}")
                            return fail("genre_filtre")
                except Exception as e:
                    if self.debug:
                        print(f"[{t:.2f}s] Erreur classification de genre: {e}")
                    return fail("genre_exception")

            # Frame validée si on arrive ici
            metrics["valid"] = True
            metrics["reason"] = ["all_pass"]  # Or clear reasons if it was an empty list
            metrics["proc_ms"] = (time.time() - start_time) * 1000
            return metrics

        except Exception as e:
            if self.debug:
                print(f"[ERROR] Exception majeure dans analyze_frame @ {t:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return fail("exception_globale")


class VideoAnalyzer:
    def __init__(
            self,
            fa: FrameAnalyzer,
            min_dur: float,
            sample_rate: float,
            refine_rate: float,  # refine_rate is not currently used in process method
            max_gap: float = 0.2,
            num_workers: int = 4
    ):
        self.fa = fa
        self.min_dur = min_dur
        self.sample_rate = sample_rate  # Frames per second for sampling
        # self.refine_rate = refine_rate # Not used
        self.max_gap_frames = int(max_gap * sample_rate)  # Max tolerated invalid frames between valid ones
        self.num_workers = num_workers

    def process(self, video_path: str, out_dir: str = "clips") -> List[Tuple[float, float]]:
        """Analyse la vidéo et retourne les intervalles (start, end) des segments valides."""
        vid_path_obj = Path(video_path)  # Renamed to avoid conflict with cv2.VideoCapture variable
        cap = cv2.VideoCapture(str(vid_path_obj))
        if not cap.isOpened():
            print(f"[ERROR] Impossible d'ouvrir la vidéo : {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # fps can be 0 if video has issues or is image sequence
            print(f"[ERROR] FPS invalide ({fps}) pour la vidéo: {video_path}. Vérifiez le fichier.")
            cap.release()
            return []

        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_video / fps if total_frames_video > 0 and fps > 0 else 0

        if duration == 0:
            print(
                f"[WARN] Durée de la vidéo nulle ou FPS/Frame count invalide pour {video_path}. Tentative de lecture frame par frame.")
            # Fallback for videos where duration cannot be determined easily (e.g. some streams or malformed files)
            # This part will be slow as it reads all frames.
            # For now, let's stick to the arange method if duration is known, otherwise, this path is problematic without rework.
            # If duration can't be trusted, np.arange won't work.
            # Consider simply returning [] if duration is 0 or negative
            cap.release()
            return []

        if self.fa.debug:
            print(
                f"[DEBUG] Vidéo ouverte: {video_path}, {fps:.2f} FPS, ~{total_frames_video} frames, durée {duration:.2f}s")

        # Timestamps for sampling based on sample_rate
        # dt is the time interval between sampled frames
        dt_sample = 1.0 / self.sample_rate
        timestamps_to_sample = list(np.arange(0.0, duration, dt_sample))

        if not timestamps_to_sample:
            print(
                f"[WARN] Aucun timestamp à échantillonner pour la vidéo {video_path} (durée: {duration:.2f}s, sample_rate: {self.sample_rate}fps).")
            cap.release()
            return []

        processed_results: List[Tuple[float, Dict[str, Any]]] = []  # Store (timestamp, metrics_dict)

        # Submit frames for analysis using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
            futures_map: Dict[Any, float] = {}  # Future -> timestamp

            # tqdm progress bar for frame submission
            pbar_submit = tqdm(total=len(timestamps_to_sample), desc=f"Soumission frames {vid_path_obj.name}",
                               disable=not self.fa.debug)

            for t_sample in timestamps_to_sample:
                cap.set(cv2.CAP_PROP_POS_MSEC, t_sample * 1000)
                ret, frame = cap.read()  # Use read() which does grab() + retrieve()
                if not ret or frame is None:
                    if self.fa.debug: print(f"[WARN] Impossible de lire la frame à {t_sample:.2f}s pour {video_path}")
                    break
                    # Submit a copy of the frame to avoid issues with shared memory if cap.read() reuses buffer
                future = exe.submit(self.fa.analyze_frame, frame.copy(), t_sample)
                futures_map[future] = t_sample
                pbar_submit.update(1)
            pbar_submit.close()

            # Collect results, maintaining order if possible (though ThreadPoolExecutor doesn't guarantee strict order of completion)
            # We will sort them by timestamp later.
            temp_results = []
            pbar_collect = tqdm(total=len(futures_map), desc=f"Analyse frames {vid_path_obj.name}",
                                disable=not self.fa.debug)
            for future in futures_map:  # Iterate over the submitted futures
                ts = futures_map[future]
                try:
                    result_metrics = future.result()  # This blocks until the future is complete
                    temp_results.append((ts, result_metrics))
                except Exception as e:
                    print(f"[ERROR][Thread] Erreur lors de l'analyse de la frame @ {ts:.2f}s : {e}")
                    # Create a dummy failed result to keep the structure
                    dummy_metrics: Dict[str, Any] = {"valid": False, "t": ts, "reason": ["thread_exception"],
                                                     "proc_ms": 0}
                    temp_results.append((ts, dummy_metrics))
                pbar_collect.update(1)
            pbar_collect.close()

            # Sort results by timestamp to ensure chronological order for segment merging
            processed_results = sorted(temp_results, key=lambda x: x[0])

        cap.release()

        # Merge valid frames into segments
        valid_intervals: List[Tuple[float, float]] = []
        current_segment_start_time: Optional[float] = None
        last_valid_frame_time: float = 0.0
        consecutive_invalid_frames_count: int = 0

        for t_frame, metrics in processed_results:
            if metrics["valid"]:
                if current_segment_start_time is None:  # Start of a new segment
                    # Start segment from this valid frame's time, or slightly before if needed (e.g., t_frame - dt_sample / 2)
                    current_segment_start_time = t_frame
                last_valid_frame_time = t_frame
                consecutive_invalid_frames_count = 0
            else:  # Frame is invalid
                if current_segment_start_time is not None:  # If we are in a segment
                    consecutive_invalid_frames_count += 1
                    # If gap is too large, end the current segment
                    if consecutive_invalid_frames_count > self.max_gap_frames:
                        segment_duration = last_valid_frame_time - current_segment_start_time
                        if segment_duration >= self.min_dur:
                            # Add dt_sample to last_valid_frame_time to include its duration
                            valid_intervals.append((current_segment_start_time, last_valid_frame_time + dt_sample))
                        current_segment_start_time = None
                        consecutive_invalid_frames_count = 0

        # After loop, check if there's an open segment
        if current_segment_start_time is not None:
            segment_duration = last_valid_frame_time - current_segment_start_time
            if segment_duration >= self.min_dur:
                valid_intervals.append((current_segment_start_time, last_valid_frame_time + dt_sample))

        # The `out_dir` and clip saving part was removed from this class as per the user's initial code.
        # `VideoAnalyzer.process` is now expected to return only the intervals.
        # The calling script (process_video.py) will handle the ffmpeg cutting.
        return valid_intervals