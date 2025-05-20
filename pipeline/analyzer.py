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
from typing import List, Tuple, Optional
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
from segmenters.skin import SkinSegmenter
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
        body_threshold: float = 50.0,
        face_threshold: float = 75.0,
        gender: str = "tous",
        enable_body: bool = True,
        enable_skin: bool = True,
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
        # Seuils et flags
        self.body_th = body_threshold
        self.face_th = face_threshold
        self.gender_target = gender.lower()
        self.debug = debug
        self.enable_body = enable_body
        self.enable_skin = enable_skin
        self.enable_face = enable_face
        self.enable_gender = enable_gender
        self.enable_nsfw = enable_nsfw
        self.nsfw_mode = nsfw_mode
        self.min_gender_conf = min_gender_confidence
        self.min_face_conf = min_face_confidence
        self.max_head_pitch = max_head_pitch
        self.max_head_yaw = max_head_yaw
        self.max_head_roll = max_head_roll

        # Choix du device (CPU/GPU)
        if torch.cuda.is_available() and not device.lower().startswith("cpu"):
            self.yolo_device = device if device.startswith("cuda") else f"cuda:{device}"
            # Index du device pour les modèles HuggingFace (CPU=-1, GPU=index)
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

        # Avertir si Mediapipe (pour pose tête) est absent
        if self.enable_face:
            import importlib.util
            if importlib.util.find_spec("mediapipe") is None:
                if self.debug:
                    print("[WARN] Module mediapipe non installé - l'orientation de la tête ne sera pas évaluée.")

        # Chargement du détecteur NSFW si demandé
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

        # Chargement du classifieur de genre si demandé
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

    def analyze_frame(self, frame: np.ndarray, t: float = 0.0) -> dict:
        """Analyse une frame vidéo à l'instant t et retourne des métriques et un drapeau valid/invalid."""
        start = time.time()
        metrics = {
            "valid": False, "t": t,
            "skin_pct": None, "mask_pct": None,
            "gender": None, "nsfw": None, "nsfw_probas": None,
            "pitch": None, "yaw": None, "roll": None,
            "proc_ms": 0.0, "reason": []
        }

        def fail(reason: str):
            metrics["valid"] = False
            metrics["reason"].append(reason)
            metrics["proc_ms"] = (time.time() - start) * 1000
            return metrics

        try:
            # 1) NSFW detection (drop SFW frames)
            if self.enable_nsfw and self.nsfw_detector:
                # call with selected sensitivity mode
                is_ns, probas = self.nsfw_detector.predict(frame, self.nsfw_mode)
                metrics["nsfw"] = is_ns
                metrics["nsfw_probas"] = probas
                if self.debug:
                    # Display all probability levels
                    prob_dict = probas[0] if isinstance(probas, (list, tuple)) else probas
                    prob_str = ", ".join(f"{lvl.name}:{score:.3f}" for lvl, score in prob_dict.items())
                    print(f"[{t:.2f}s] NSFW probabilities ({self.nsfw_mode}): {prob_str}")
                    print(f"[{t:.2f}s] is_nsfw -> {is_ns}")
                # Exclude non-NSFW frames
                if not is_ns:
                    return fail("nsfw")

            # 2) Détection du corps (segmentation personnes)
            if self.enable_body and self.person_model is not None:
                res = self.person_model.predict(frame, verbose=False, device=self.yolo_device)
                if not res or res[0].masks is None or res[0].masks.data.size(0) == 0:
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun corps détecté")
                    return fail("personne")
                # Couverture de la personne détectée (masque)
                mask = res[0].masks.data[0].cpu().numpy()
                body_area = mask.sum() / mask.size * 100
                if body_area < self.body_th:
                    if self.debug:
                        print(f"[{t:.2f}s] Peau visible {body_area:.1f}% < seuil {self.body_th}%")
                    return fail("corps")

            # 3) Détection visage (bbox)
            if self.enable_face and self.face_model is not None:
                res = self.face_model.predict(frame, verbose=False, device=self.yolo_device)
                if not res or res[0].boxes.data.size(0) == 0:
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun visage détecté")
                    return fail("visage")
                # Visage détecté => vérifier le pourcentage de visage visible (sans masque)
                x1, y1, x2, y2 = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
                face_roi = frame[y1:y2, x1:x2]  # découpe explicite ici
                mask_pct = face_mask_percentage(face_roi)
                metrics["mask_pct"] = mask_pct
                if mask_pct > (100 - self.face_th):
                    if self.debug:
                        print(f"[{t:.2f}s] Visage trop masqué ({100-mask_pct:.1f}% visible)")
                    return fail("masque")

            # 4) Détection surface de peau
            if self.enable_skin and self.skin_model is not None:
                skin_mask, skin_pct = self.skin_model.detect(frame)
                metrics["skin_pct"] = skin_pct
                if skin_pct < self.min_face_conf:
                    if self.debug:
                        # self.min_face_conf est une fraction (ex: 0.5 pour 50%)
                        print(f"[{t:.2f}s] Peau visible {skin_pct:.1f}% < seuil {self.min_face_conf*100:.0f}%")
                    return fail("peau")

            # 5) Orientation de la tête (si visage détecté)
            if self.enable_face:
                try:
                    angles = estimate_head_pose(frame)
                    metrics["pitch"], metrics["yaw"], metrics["roll"] = angles
                    if abs(metrics["pitch"]) > self.max_head_pitch or abs(metrics["yaw"]) > self.max_head_yaw or abs(metrics["roll"]) > self.max_head_roll:
                        if self.debug:
                            print(f"[{t:.2f}s] Tête hors limites ({metrics['pitch']:.1f},{metrics['yaw']:.1f},{metrics['roll']:.1f})")
                        return fail("pose")
                except Exception:
                    # Échec silencieux si absence de mediapipe par ex.
                    pass

            # 6) Classification de genre (si demandé)
            if self.enable_gender and self.gender_clf is not None:
                try:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    preds = self.gender_clf(img)[0]
                    label = preds["label"]
                    conf = preds["score"]
                    if conf < 0.5:
                        if self.debug:
                            print(f"[{t:.2f}s] Confiance genre {conf:.2f} < 0.5")
                        return fail("genre_conf")
                    label = normalize_gender(label)
                    metrics["gender"] = label
                    if self.gender_target in ("male", "female"):
                        # Filtre homme/femme demandé => exclure l'autre genre
                        if label != self.gender_target:
                            if self.debug:
                                print(f"[{t:.2f}s] Genre détecté {label} != filtre {self.gender_target}")
                            return fail("genre")
                except Exception as e:
                    if self.debug:
                        print(f"[ERROR] analyze_frame exception: {e}")
                    return fail("exception")

            # Frame validée si on arrive ici
            metrics["valid"] = True
            metrics["reason"] = ["all_pass"]
            metrics["proc_ms"] = (time.time() - start) * 1000
            return metrics

        except Exception as e:
            if self.debug:
                print(f"[ERROR] analyze_frame exception: {e}")
            return fail("exception")

class VideoAnalyzer:
    def __init__(
        self,
        fa: FrameAnalyzer,
        min_dur: float,
        sample_rate: float,
        refine_rate: float,
        max_gap: float = 0.2,
        num_workers: int = 4
    ):
        self.fa = fa
        self.min_dur = min_dur
        self.sample_rate = sample_rate
        self.refine_rate = refine_rate
        self.max_gap_frames = int(max_gap * sample_rate)
        self.num_workers = num_workers

    def process(self, video_path: str, out_dir: str = "clips") -> List[Tuple[float, float]]:
        """Analyse la vidéo et génère les clips extraits, retourne les intervalles (start, end)."""
        vid = Path(video_path)
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[ERROR] Impossible d'ouvrir la vidéo : {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"[ERROR] FPS invalide ({fps}). Vérifiez le fichier vidéo.")
            cap.release()
            return []
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        if self.fa.debug:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[DEBUG] Vidéo ouverte: {fps:.2f} FPS, ~{total_frames} frames, durée {duration:.2f}s")
        dt = 1.0 / self.sample_rate

        intervals = []
        in_seg = False
        gaps = 0
        last_t = 0.0
        timestamps, futures = [], []
        # Soumettre les frames pour analyse
        with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
            for t in np.arange(0.0, duration, dt):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                if not cap.grab():
                    break
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break
                futures.append(exe.submit(self.fa.analyze_frame, frame.copy(), t))
                timestamps.append(t)
        cap.release()

        results = []
        for fut, ts in zip(futures, timestamps):
            try:
                results.append((ts, fut.result()))
            except Exception as e:
                print(f"[Thread] Erreur @ {ts:.2f}s : {e}")

        intervals = []
        in_seg = False
        gaps = 0
        last_t = 0.0
        seg_start = 0.0
        for ts, info in results:
            if info["valid"]:
                if not in_seg:
                    seg_start = max(0.0, ts - dt)
                    in_seg = True
                last_t = ts
                gaps = 0
            elif in_seg:
                gaps += 1
                if gaps > self.max_gap_frames:
                    if last_t - seg_start >= self.min_dur:
                        intervals.append((seg_start, last_t))
                    in_seg = False
        # Si la vidéo se termine alors qu'un segment est en cours
        if in_seg and last_t - seg_start >= self.min_dur:
            intervals.append((seg_start, last_t))

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        saved_intervals = []
        prefix_base = vid.stem or "video"
        prefix = "".join(c for c in prefix_base if c.isalnum() or c in (" ", "_", "-")).strip() or "video"
        for i, (s, e) in enumerate(intervals, start=1):
            out_file = Path(out_dir) / f"{prefix}_edited_{i:03d}.mp4"
            try:
                if self.fa.debug:
                    print(f"[DEBUG] Découpage clip {i:03d} : {s:.2f}s -> {e:.2f}s")
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", f"{s:.2f}", "-to", f"{e:.2f}",
                    "-c", "copy", str(out_file)
                ], check=True)
                saved_intervals.append((s, e))
            except FileNotFoundError:
                # ffmpeg introuvable : interrompre le traitement
                raise RuntimeError("Commande 'ffmpeg' introuvable. Assurez-vous que FFmpeg est installé.")
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] Échec de la découpe pour le segment {i} ({s:.2f}-{e:.2f}s) : {exc}")
                # Segment non généré (on passe au suivant)
                continue

        return saved_intervals
