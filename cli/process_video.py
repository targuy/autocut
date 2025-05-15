#!/usr/bin/env python3
# cli/process_video.py

import argparse
import time
from config import load_config
from pipeline.analyzer import FrameAnalyzer, VideoAnalyzer

def main():
    parser = argparse.ArgumentParser(description="AutoCutVideo – Analyse de vidéo et découpe intelligente")
    parser.add_argument("--config", "-c", default="config.yaml", help="Chemin du fichier YAML")
    parser.add_argument("--debug", action="store_true", help="Activer le mode debug (prioritaire sur YAML)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug = True

    print("=== AutoCutVideo Debut ===")
    print(f"Video            : {cfg.input_video}")
    print(f"Sortie clips     : {cfg.output_dir}")
    print(f"Device           : {cfg.device}")
    print(f"Filtre genre     : {cfg.gender_filter}")
    print(f"Debug            : {cfg.debug}")
    print(f"Détection genre  : {cfg.enable_gender_detection}")
    print(f"Détection visage : {cfg.enable_face_detection}")
    print(f"Détection peau   : {cfg.enable_skin_detection}")
    print(f"Détection corps  : {cfg.enable_body_detection}")
    print(f"Détection NSFW   : {cfg.enable_nsfw} (mode={cfg.nsfw_mode})")
    print(f"Seuil pose       : pitch={cfg.max_head_pitch} yaw={cfg.max_head_yaw} roll={cfg.max_head_roll}")
    print("-------------------------------")

    fa = FrameAnalyzer(
        person_segm_weights   = cfg.person_segm_weights,
        face_bbox_weights     = cfg.face_bbox_weights,
        skin_segm_weights     = cfg.skin_segm_weights,
        gender_model_id       = cfg.gender_model_id,
        device                = cfg.device,
        body_threshold        = cfg.min_skin_percentage,
        face_threshold        = 100 - cfg.max_face_mask_percentage,
        gender                = cfg.gender_filter,
        debug                 = cfg.debug,
        enable_body           = cfg.enable_body_detection,
        enable_skin           = cfg.enable_skin_detection,
        enable_face           = cfg.enable_face_detection,
        enable_gender         = cfg.enable_gender_detection,
        enable_nsfw           = cfg.enable_nsfw,
        nsfw_mode             = cfg.nsfw_mode,
        min_gender_confidence = cfg.min_gender_confidence,
        min_face_confidence   = cfg.min_face_confidence,
        max_head_pitch        = cfg.max_head_pitch,
        max_head_yaw          = cfg.max_head_yaw,
        max_head_roll         = cfg.max_head_roll,
    )

    va = VideoAnalyzer(
        fa,
        min_dur     = cfg.min_clip_duration,
        sample_rate = cfg.sample_rate,
        refine_rate = cfg.refine_rate,
        num_workers = cfg.num_workers,
        max_gap     = cfg.max_gap
    )

    start = time.time()
    clips = va.process(cfg.input_video, out_dir=cfg.output_dir)
    elapsed = time.time() - start

    total_clip_dur = sum(e - s for s, e in clips)
    print("-------------------------------")
    print(f"Analyse terminée en {elapsed:.2f} secondes")
    print(f"Clips générés   : {len(clips)}")
    print(f"Durée totale    : {total_clip_dur:.2f} sec")
    if clips:
        print("Clips découpés  :")
        for i, (s, e) in enumerate(clips, 1):
            print(f"  [{i:03d}] {s:.2f}s → {e:.2f}s")
    else:
        print("Aucun segment valide trouvé.")
    print("=== AutoCutVideo Fin ===")

if __name__ == "__main__":
    main()
