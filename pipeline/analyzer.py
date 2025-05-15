import cv2
import subprocess
from pathlib import Path
from typing import List, Tuple
import time
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import snapshot_download
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    pipeline as hf_pipeline,
)
from PIL import Image
from segmenters.skin import SkinSegmenter
from detectors.mask import face_mask_percentage
from detectors.nsfw import NSFWWrapper
from detectors.pose import estimate_head_pose
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def normalize_gender(label: str) -> str:
    label = label.lower()
    if "female" in label or "woman" in label:
        return "female"
    elif "homme" in label or "man" in label:
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

        if torch.cuda.is_available():
            self.yolo_device = device if device.startswith("cuda") else f"cuda:{device}"
            self.hf_device = int(device.split(":")[-1]) if ":" in device else (int(device) if device.isdigit() else 0)
        else:
            self.yolo_device = "cpu"
            self.hf_device = -1

        self.person_model = YOLO(person_segm_weights, task="segment") if enable_body else None
        self.face_model = YOLO(face_bbox_weights, task="detect") if enable_face else None
        self.skin_model = SkinSegmenter(skin_segm_weights, device=self.yolo_device) if enable_skin else None
        self.nsfw_detector = NSFWWrapper(device=self.yolo_device, dtype="bfloat16") if enable_nsfw else None

        if enable_gender:
            model_dir = snapshot_download(repo_id=gender_model_id)
            self.gender_clf = hf_pipeline(
                task="image-classification",
                model=AutoModelForImageClassification.from_pretrained(model_dir),
                feature_extractor=AutoFeatureExtractor.from_pretrained(model_dir),
                device=self.hf_device,
            )

    def analyze_frame(self, frame: np.ndarray, t: float = 0.0) -> dict:
        start = time.time()
        metrics = {
            "valid": False, "t": t,
            "skin_pct": None, "mask_pct": None,
            "gender": None, "nsfw": None, "nsfw_probas": None,
            "pitch": None, "yaw": None, "roll": None,
            "proc_ms": 0.0, "reason": []
        }

        def fail(reason):
            metrics["valid"] = False
            metrics["reason"].append(reason)
            metrics["proc_ms"] = (time.time() - start) * 1000
            return metrics

        # NSFW
        if self.enable_nsfw and self.nsfw_detector:
            is_ns, probas = self.nsfw_detector.predict(frame, self.nsfw_mode)
            metrics["nsfw"] = is_ns
            metrics["nsfw_probas"] = probas
            if not is_ns:
                return fail("nsfw")

        # BODY
        if self.enable_body and self.person_model:
            res_p = self.person_model(frame, device=self.yolo_device)[0]
            if not (hasattr(res_p, "masks") and res_p.masks and len(res_p.masks.data) > 0):
                return fail("body")
            mask = res_p.masks.data[0].cpu().numpy() > 0.5
            ys, xs = np.where(mask)
            y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
            body = frame[y1:y2, x1:x2]
            if self.enable_skin:
                skin_pct = self.skin_model.percentage(body)
                metrics["skin_pct"] = skin_pct
                if skin_pct < self.body_th:
                    return fail("skin")

        # HEAD
        if self.enable_face and self.face_model:
            res_f = self.face_model(frame, device=self.yolo_device, conf=self.min_face_conf)[0]
            if not (hasattr(res_f, "boxes") and len(res_f.boxes) > 0):
                return fail("face")
            bx1, by1, bx2, by2 = res_f.boxes.xyxy[0].cpu().numpy().astype(int)
            face = frame[by1:by2, bx1:bx2]
            pose = estimate_head_pose(frame, (bx1, by1, bx2, by2))
            if pose is None:
                return fail("pose_estimation")
            pitch, yaw, roll = pose
            metrics.update({"pitch": pitch, "yaw": yaw, "roll": roll})
            if abs(pitch) > self.max_head_pitch or abs(yaw) > self.max_head_yaw or abs(roll) > self.max_head_roll:
                if self.debug:
                    print(f"[{t:.2f}s] HEAD: pose angles out of range (pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f})")
                return fail("pose_angle")

            mask_pct = face_mask_percentage(face)
            metrics["mask_pct"] = mask_pct
            if 100 - mask_pct < self.face_th:
                return fail("mask")

            if self.enable_gender:
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                lab = self.gender_clf(pil)[0]
                norm = normalize_gender(lab["label"])
                metrics["gender"] = norm
                if lab["score"] < self.min_gender_conf:
                    return fail("gender_conf")
                if self.gender_target != "tous" and norm != self.gender_target:
                    return fail("gender")

        metrics["valid"] = True
        metrics["reason"] = "all_pass"
        metrics["proc_ms"] = (time.time() - start) * 1000
        return metrics


class VideoAnalyzer:
    def __init__(self, fa: FrameAnalyzer, min_dur: float, sample_rate: float, refine_rate: float, max_gap: float = 0.2, num_workers: int = 4):
        self.fa = fa
        self.min_dur = min_dur
        self.sample_rate = sample_rate
        self.refine_rate = refine_rate
        self.max_gap_frames = int(max_gap * sample_rate)
        self.num_workers = num_workers

    def process(self, video_path: str, out_dir: str = "clips") -> List[Tuple[float, float]]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        dt = 1.0 / self.sample_rate

        intervals = []
        in_seg = False
        gaps = 0
        last_valid_t = 0
        idx = 0
        t = 0.0
        total = int(duration * self.sample_rate)

        print(f"[Processing] total={total} frames, max_gap_frames={self.max_gap_frames}")
        futures = []
        timestamps = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for t in tqdm(np.arange(0.0, duration, dt), desc="Submitting frames"):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                frame_grabbed = cap.grab()
                if not frame_grabbed:
                    break
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break
                futures.append(executor.submit(self.fa.analyze_frame, frame.copy(), t))
                timestamps.append(t)
                t += dt

            results = []
            for i, fut in enumerate(tqdm(futures, desc="Analyzing frames")):
                try:
                    result = fut.result()
                    results.append((timestamps[i], result))
                except Exception as e:
                    print(f"[Thread error @ t={timestamps[i]:.2f}s]: {e}")

        for t, info in results:
            idx += 1
            if info["valid"]:
                if not in_seg:
                    seg_start = max(0.0, t - dt)
                    in_seg = True
                last_valid_t = t
                gaps = 0
            elif in_seg:
                gaps += 1
                if gaps > self.max_gap_frames:
                    seg_end = last_valid_t
                    if seg_end - seg_start >= self.min_dur:
                        intervals.append((seg_start, seg_end))
                    in_seg = False
            if self.fa.debug:
                print(f"[{idx}/{total}] t={t:.2f}s valid={info['valid']} reason={info['reason']}")

        if in_seg and (last_valid_t - seg_start) >= self.min_dur:
            intervals.append((seg_start, last_valid_t))
        cap.release()

        Path(out_dir).mkdir(exist_ok=True)
        with tqdm(total=len(futures), desc="Analyzing frames") as pbar:
            for i, fut in enumerate(futures):
                try:
                    result = fut.result()
                    results.append((timestamps[i], result))
                except Exception as e:
                    print(f"[Thread error @ t={timestamps[i]:.2f}s]: {e}")
                finally:
                    pbar.update(1)
        return intervals
