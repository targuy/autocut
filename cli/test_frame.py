#!/usr/bin/env python3
"""
cli/test_frame.py
Test complet de FrameAnalyzer sur une seule image, avec sélection de branche (head, full)
Usage :
  python -m cli.test_frame IMAGE_PATH --config CONFIG_YAML [--branch full|head]
"""

import argparse
import json
import cv2
from config import load_config
from pipeline.analyzer import FrameAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Test une image via FrameAnalyzer")
    parser.add_argument("image", help="Chemin vers l'image à tester")
    parser.add_argument("--config", default="config.yml", help="Fichier de configuration YAML")
    parser.add_argument("--branch", choices=["full", "head"], default="full", help="Branche à tester")
    parser.add_argument("--debug", action="store_true", help="Activer les logs détaillés (debug)")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug = True

    enable_face = True
    enable_gender = (args.branch == "full") and cfg.enable_gender_detection

    fa = FrameAnalyzer(
        face_bbox_weights=cfg.face_bbox_weights,
        gender_model_id=cfg.gender_model_id,
        device=cfg.device,
        min_visible_face_threshold=(100 - cfg.max_face_mask_percentage),
        min_face_bbox_area_pct=cfg.min_face_bbox_area_pct,
        gender=cfg.gender_filter,
        enable_face=enable_face,
        enable_gender=enable_gender,
        enable_nsfw=cfg.enable_nsfw,
        nsfw_mode=cfg.nsfw_mode,
        min_gender_confidence=cfg.min_gender_confidence,
        min_face_confidence=cfg.min_face_confidence,
        max_head_pitch=cfg.max_head_pitch,
        max_head_yaw=cfg.max_head_yaw,
        max_head_roll=cfg.max_head_roll,
        debug=cfg.debug,
    )

    img = cv2.imread(args.image)
    if img is None:
        print(json.dumps({"error": "Image non lisible"}, ensure_ascii=False))
        return

    res = fa.analyze_frame(img, t=0.0)
    out = {
        "t": res.get("t"),
        "valid": res.get("valid"),
        "mask_pct": res.get("mask_pct"),
        "face_visible_pct": res.get("face_visible_pct"),
        "gender": res.get("gender"),
        "gender_conf": res.get("gender_conf"),
        "pitch": res.get("pitch"),
        "yaw": res.get("yaw"),
        "roll": res.get("roll"),
        "proc_ms": res.get("proc_ms"),
        "reason": res.get("reason"),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
