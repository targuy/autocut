import cv2
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import time
import numpy as np
import torch

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

from PIL import Image
from segmenters.skin import SkinSegmenter
from detectors.mask import face_mask_percentage
from detectors.nsfw import NSFWWrapper
from detectors.pose import estimate_head_pose
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
                print("[WARN] Module mediapipe non installé - l'orientation de la tête ne sera pas évaluée.")

        # Chargement du détecteur NSFW si demandé
        if enable_nsfw:
            import importlib.util
            if importlib.util.find_spec("nsfw_image_detector") is None:
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
            # 1) Détection NSFW
            if self.enable_nsfw and self.nsfw_detector:
                is_ns, probas = self.nsfw_detector.predict(frame, self.nsfw_mode)
                metrics["nsfw"], metrics["nsfw_probas"] = is_ns, probas
                if not is_ns:
                    if self.debug:
                        print(f"[{t:.2f}s] Frame ignoré (contenu non NSFW)")
                    return fail("nsfw")

            # 2) Détection corps + peau
            if self.enable_body and self.person_model:
                res_p = self.person_model(frame, device=self.yolo_device)[0]
                if not (hasattr(res_p, "masks") and res_p.masks and res_p.masks.data.shape[0] > 0):
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun corps détecté")
                    return fail("body")
                mask = res_p.masks.data[0].cpu().numpy() > 0.5
                ys, xs = np.where(mask)
                body = frame[ys.min():ys.max(), xs.min():xs.max()]
                if self.enable_skin:
                    skin_pct = self.skin_model.percentage(body)
                    metrics["skin_pct"] = skin_pct
                    if skin_pct < self.body_th:
                        if self.debug:
                            print(f"[{t:.2f}s] Frame ignoré (peau visible {skin_pct:.1f}% < seuil {self.body_th}%)")
                        return fail("skin")

            # 3) Tête + visage + genre
            if self.enable_face and self.face_model:
                res_f = self.face_model(frame, device=self.yolo_device, conf=self.min_face_conf)[0]
                if not (hasattr(res_f, "boxes") and len(res_f.boxes) > 0):
                    if self.debug:
                        print(f"[{t:.2f}s] Aucun visage détecté")
                    return fail("face")
                bx1, by1, bx2, by2 = res_f.boxes.xyxy[0].cpu().numpy().astype(int)
                face = frame[by1:by2, bx1:bx2]

                # Orientation de la tête
                pose = estimate_head_pose(frame, (bx1, by1, bx2, by2))
                if pose is None:
                    if self.debug:
                        print(f"[{t:.2f}s] Pose de tête non estimée")
                    return fail("pose_estimation")
                pitch, yaw, roll = pose
                metrics.update({"pitch": pitch, "yaw": yaw, "roll": roll})
                if abs(pitch) > self.max_head_pitch or abs(yaw) > self.max_head_yaw or abs(roll) > self.max_head_roll:
                    if self.debug:
                        print(f"[{t:.2f}s] Tête: angles hors limites (pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f})")
                    return fail("pose_angle")

                # Pourcentage de visage non masqué
                mask_pct = face_mask_percentage(face)
                metrics["mask_pct"] = mask_pct
                if 100 - mask_pct < self.face_th:
                    if self.debug:
                        print(f"[{t:.2f}s] Visage visible à {100 - mask_pct:.1f}% (< seuil {self.face_th}%) -> frame ignoré")
                    return fail("mask")

                # Détermination du genre
                if self.enable_gender:
                    pil_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    lab = self.gender_clf(pil_img)[0] if self.gender_clf else {"label": None, "score": 0.0}
                    norm = normalize_gender(lab["label"] if lab["label"] else "")
                    metrics["gender"] = norm
                    if lab["score"] < self.min_gender_conf:
                        if self.debug:
                            print(f"[{t:.2f}s] Confiance genre insuffisante ({lab['score']:.2f} < {self.min_gender_conf})")
                        return fail("gender_conf")
                    if self.gender_target != "tous" and norm != self.gender_target:
                        if self.debug:
                            target_label = "femme" if self.gender_target == "female" else ("homme" if self.gender_target == "male" else self.gender_target)
                            norm_label = "femme" if norm == "female" else ("homme" if norm == "male" else norm)
                            print(f"[{t:.2f}s] Genre détecté : {norm_label} (filtre demandé : {target_label}) -> frame ignoré")
                        return fail("gender")

            # Si toutes les vérifications sont passées, la frame est valide
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
        """Analyse la vidéo donnée et génère les clips extraits, puis retourne les intervalles (début, fin) de chaque clip."""
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
        # Infos vidéo pour debug
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

        # Soumission des frames à analyser (multithread)
        with ThreadPoolExecutor(max_workers=self.num_workers) as exe:
            for t in tqdm(np.arange(0.0, duration, dt), desc="Analyse frames"):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                grabbed = cap.grab()
                if not grabbed:
                    break
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break
                futures.append(exe.submit(self.fa.analyze_frame, frame.copy(), t))
                timestamps.append(t)
        cap.release()

        # Collecte des résultats d'analyse
        results = []
        for fut, t in zip(futures, timestamps):
            try:
                results.append((t, fut.result()))
            except Exception as e:
                print(f"[Thread] erreur @ t={t:.2f}s : {e}")

        # Découpage en segments en se basant sur la validité des frames
        for t, info in results:
            if info["valid"]:
                if not in_seg:
                    seg_start = max(0.0, t - dt)
                    in_seg = True
                last_t = t
                gaps = 0
            elif in_seg:
                gaps += 1
                if gaps > self.max_gap_frames:
                    if last_t - seg_start >= self.min_dur:
                        intervals.append((seg_start, last_t))
                    in_seg = False

        # Clôturer le segment final éventuellement ouvert
        if in_seg and last_t - seg_start >= self.min_dur:
            intervals.append((seg_start, last_t))

        # Écriture des clips vidéo via ffmpeg
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        saved_intervals: List[Tuple[float, float]] = []
        # Préfixe pour nommer les fichiers de sortie (basé sur nom de la vidéo source)
        prefix_base = Path(video_path).stem
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
