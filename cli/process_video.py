#!/usr/bin/env python3
# cli/process_video.py
"""
Point d’entrée principal AutoCutVideo
– Analyse vidéo
– Découpe des clips
– (Optionnel) génération de titres via LM-Studio
"""
import argparse
import time
from pathlib import Path
from typing import List, Tuple
from typing import Optional, Callable

from tqdm import tqdm

from config import load_config
from pipeline.analyzer import FrameAnalyzer, VideoAnalyzer

# --- ajout : générateur de titres multimodal ------------------------------ #
try:
    from title_generator import TitleGenerator as _TG
except ImportError:
    _TG = None  # laisser None si module manquant
# -------------------------------------------------------------------------- #

def _build_title_generator(cfg) -> Optional[Callable]:
    """
    Retourne une instance de TitleGenerator si :
      • activé dans le YAML
      • module disponible
    Sinon retourne None.
    """
    if not cfg.title_generation.enabled:
        return None

    if _TG is None:
        print("[WARN] title_generation activé mais le module title_generator.py est introuvable.")
        return None

    # PyCharm comprendra maintenant que _TG est Callable
    return _TG(
        endpoint=cfg.title_generation.endpoint,
        model=cfg.title_generation.model,
        prompt_template=cfg.title_generation.prompt,
    )
# ────────────────────────────────────────────────────────────────────────────── #

def _print_header(cfg):
    print("=== AutoCutVideo Début ===")
    print(f"Vidéo            : {cfg.input_video}")
    print(f"Sortie clips     : {cfg.output_dir}")
    print(f"Device           : {cfg.device}")
    print(f"Workers          : {cfg.num_workers}")
    print(f"Filtre genre     : {cfg.gender_filter}")
    print(f"Debug            : {cfg.debug}")
    print(f"Modules actifs   : body={cfg.enable_body_detection} "
          f"skin={cfg.enable_skin_detection} face={cfg.enable_face_detection} "
          f"gender={cfg.enable_gender_detection} nsfw={cfg.enable_nsfw}")
    print(f"NSFW mode        : {cfg.nsfw_mode}")
    print(f"Pose seuils      : pitch={cfg.max_head_pitch}  "
          f"yaw={cfg.max_head_yaw}  roll={cfg.max_head_roll}")
    if cfg.title_generation.enabled:
        print("Génération titre : ON  "
              f"(modèle={cfg.title_generation.model})")
    else:
        print("Génération titre : OFF")
    print("-" * 31)


def _rename_clip(path: Path, title: str) -> Path:
    """Renomme `path` avec `title` (gestion doublons)."""
    base = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()
    stem, ext = base, path.suffix
    candidate = path.with_stem(stem)

    counter = 1
    while candidate.exists():
        candidate = path.with_stem(f"{stem}_{counter:02d}")
        counter += 1
    path.rename(candidate)
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="AutoCutVideo – analyse, découpe et titrage automatique")
    parser.add_argument(
        "--config", "-c", default="config.yaml",
        help="Chemin du fichier YAML")
    parser.add_argument(
        "--debug", action="store_true",
        help="Activer le mode debug (prioritaire sur YAML)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug = True

    _print_header(cfg)

    # ---------------- Analyseur de frames ---------------- #
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
        enable_body=cfg.enable_body_detection,
        enable_skin=cfg.enable_skin_detection,
        enable_face=cfg.enable_face_detection,
        enable_gender=cfg.enable_gender_detection,
        enable_nsfw=cfg.enable_nsfw,
        nsfw_mode=cfg.nsfw_mode,
        min_gender_confidence=cfg.min_gender_confidence,
        min_face_confidence=cfg.min_face_confidence,
        max_head_pitch=cfg.max_head_pitch,
        max_head_yaw=cfg.max_head_yaw,
        max_head_roll=cfg.max_head_roll,
    )

    # ---------------- Analyseur vidéo -------------------- #
    va = VideoAnalyzer(
        fa,
        min_dur=cfg.min_clip_duration,
        sample_rate=cfg.sample_rate,
        refine_rate=cfg.refine_rate,
        num_workers=cfg.num_workers,
        max_gap=cfg.max_gap,
    )

    t0 = time.time()
    clips: List[Tuple[float, float]] = va.process(
        cfg.input_video, out_dir=cfg.output_dir)
    elapsed = time.time() - t0

    # ---------------- Titres automatiques ---------------- #
    # --------------------------------------------------------------------------- #
    title_gen = _build_title_generator(cfg)
    # --------------------------------------------------------------------------- #

#    title_gen = None
    if cfg.title_generation.enabled:
        if TitleGenerator is None:
            print("[WARN] title_generation activé mais module introuvable.")
        else:
            title_gen = TitleGenerator(
                endpoint=cfg.title_generation.endpoint,
                model=cfg.title_generation.model,
                prompt_template=cfg.title_generation.prompt,
            )

    print("-" * 31)
    print(f"Analyse terminée : {elapsed:.2f}s")
    print(f"Clips créés      : {len(clips)}")

    if not clips:
        print("Aucun segment valide.")
        print("=== Fin ===")
        return

    clip_paths: List[Path] = []
    for i, (s, e) in enumerate(clips, 1):
        path = Path(cfg.output_dir) / f"clip_{i:03d}.mp4"
        clip_paths.append(path)

    # Génération des titres si demandé
    if title_gen:
        print("Génération des titres…")
        for p in tqdm(clip_paths, desc="Titres"):
            try:
                # Produit un titre à partir d'une frame médiane
                title = title_gen.generate_title_from_video(str(p))
                new_path = _rename_clip(p, title)
                if cfg.debug:
                    print(f"{p.name} -> {new_path.name}")
            except Exception as exc:
                print(f"[TitleGen] Erreur sur {p.name}: {exc}")

    # Résumé final
    print("Clips finaux :")
    for p in Path(cfg.output_dir).iterdir():
        if p.suffix.lower() == ".mp4":
            print(f"  • {p.name}")
    print("=== AutoCutVideo Fin ===")


if __name__ == "__main__":
    main()
