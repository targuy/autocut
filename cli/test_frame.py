
#!/usr/bin/env python3
"""
cli/test_frame.py
Test complet de FrameAnalyzer sur une seule image, avec sélection de branche (head, body, full)
Usage :
  python -m cli.test_frame IMAGE_PATH --config CONFIG_YAML [--branch full|head|body]
"""

import argparse
import json
import cv2
from config import load_config
from pipeline.analyzer import FrameAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Test une image via FrameAnalyzer")
    parser.add_argument("image", help="Chemin vers l'image à tester")
    parser.add_argument("--config", default="config.yaml", help="Fichier de configuration YAML")
    parser.add_argument("--branch", choices=["full", "head", "body"], default="full", help="Branche à tester")
    parser.add_argument("--debug", action="store_true", help="Activer les logs détaillés (debug)")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Activer ou désactiver les branches en fonction de --branch
    enable_body = args.branch in ["full", "body"]
    enable_skin = enable_body and cfg.enable_skin_detection
    enable_face = args.branch in ["full", "head"]
    enable_gender = enable_face and cfg.enable_gender_detection

    fa = FrameAnalyzer(
        person_segm_weights=cfg.person_segm_weights,
        face_bbox_weights=cfg.face_bbox_weights,
        skin_segm_weights=cfg.skin_segm_weights,
        gender_model_id=cfg.gender_model_id,
        device=cfg.device,
        body_threshold=cfg.min_skin_percentage,
        face_threshold=100 - cfg.max_face_mask_percentage,
        gender=cfg.gender_filter,
        debug=cfg.debug,
        enable_body=enable_body,
        enable_skin=enable_skin,
        enable_face=enable_face,
        enable_gender=enable_gender,
        min_gender_confidence=cfg.min_gender_confidence,
        min_face_confidence=cfg.min_face_confidence,
    )

    img = cv2.imread(args.image)
    if img is None:
        print(json.dumps({"error": "Image non lisible"}))
        return

    res = fa.analyze_frame(img, t=0.0)
    out = {
        "t": res["t"],
        "valid": res["valid"],
        "skin_pct": res["skin_pct"],
        "mask_pct": res["mask_pct"],
        "gender": res["gender"],
        "proc_ms": res["proc_ms"],
    }

    detail = {}

    if res["mask_pct"] is not None:
        detail["visible_face_pct"] = 100.0 - res["mask_pct"]
    if hasattr(fa, "gender_clf") and res["gender"]:
        label = fa.gender_clf(img)[0]
        detail["gender_score"] = round(label["score"], 3)

    if res["valid"]:
        if enable_face:
            out["reason"] = "valid via HEAD"
        elif enable_body:
            out["reason"] = "valid via BODY"
    else:
        out["reason"] = "not valid"

    out["detail"] = detail
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
