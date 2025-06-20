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
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import subprocess

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Module 'tqdm' non trouvé. Veuillez installer tqdm.")
    sys.exit(1)

try:
    from config import load_config
except ImportError as e:
    print(f"[ERROR] Importation de config impossible : {e}")
    sys.exit(1)
try:
    from pipeline.analyzer import FrameAnalyzer, VideoAnalyzer
except ImportError as e:
    print(f"[ERROR] Impossible d'importer 'pipeline.analyzer' : {e}")
    sys.exit(1)

# --- ajout : générateur de titres multimodal ------------------------------ #
try:
    from pipeline.title_generator import TitleGenerator as _TG
except ImportError:
    _TG = None  # None si module manquant (pas de génération de titres)
# -------------------------------------------------------------------------- #

def _build_title_generator(cfg) -> Optional[Callable]:
    """
    Retourne une instance de TitleGenerator si :
      • activé dans le YAML
      • module disponible et LM-Studio opérationnel
    Sinon retourne None.
    """
    if not cfg.title_generation.enabled:
        return None
    if _TG is None:
        if cfg.debug:
            print("[WARN] title_generation activé mais le module title_generator.py est introuvable.")
        return None
    # Vérifier que le serveur LMStudio est lancé et le modèle chargé
    from pipeline.title_generator import ensure_server_running
    if not ensure_server_running(cfg.title_generation.model, cfg.title_generation.endpoint):
        print("[TitleGeneration] LMStudio ne répond pas. Veuillez le lancer manuellement.")
        return None
    # Retourner une fabrique configurée de TitleGenerator
    def factory():
        return _TG(
            endpoint=cfg.title_generation.endpoint,
            model=cfg.title_generation.model,
            prompt_template=cfg.title_generation.prompt,
        )
    return factory

def _print_header(cfg):
    print("=== AutoCutVideo Début ===")
    if Path(cfg.input_video).is_dir():
        print(f"Répertoire source: {cfg.input_video}")
    else:
        print(f"Vidéo            : {cfg.input_video}")
    print(f"Sortie clips     : {cfg.output_dir}")
    print(f"Device           : {cfg.device}")
    print(f"Workers          : {cfg.num_workers}")
    print(f"Filtre genre     : {cfg.gender_filter}")
    print(f"Debug            : {cfg.debug}")
    print(f"Modules actifs   : body={cfg.enable_body_detection} skin={cfg.enable_skin_detection} face={cfg.enable_face_detection} gender={cfg.enable_gender_detection} nsfw={cfg.enable_nsfw}")
    print(f"NSFW mode        : {cfg.nsfw_mode}")
    print(f"Pose seuils      : pitch={cfg.max_head_pitch}  yaw={cfg.max_head_yaw}  roll={cfg.max_head_roll}")
    if cfg.title_generation.enabled:
        print("Génération titre : ON  " f"(modèle={cfg.title_generation.model})")
    else:
        print("Génération titre : OFF")
    print("-" * 31)

def _rename_clip(path: Path, title: str) -> Path:
    """Renomme le fichier `path` en utilisant `title` (en gérant les doublons)."""
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
        description="AutoCutVideo – analyse, découpe et titrage automatique"
    )
    parser.add_argument(
        "--config", "-c", default="config.yml",
        help="Chemin du fichier YAML de configuration"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Activer le mode debug (prioritaire sur YAML)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug = True

    if cfg.debug:
        _print_header(cfg)

    input_path = Path(cfg.input_video)
    videos_to_process: List[Path] = []
    if input_path.is_dir():
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".m2ts"}
        videos_to_process = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in video_extensions]
        videos_to_process.sort()
        if not videos_to_process:
            print(f"[ERROR] Aucune vidéo trouvée dans le dossier d'entrée : {cfg.input_video}")
            sys.exit(1)
        if cfg.debug:
            print(f"Vidéos détectées : {len(videos_to_process)}")
    else:
        videos_to_process = [input_path]

    # ---------------- Analyseur de frames ---------------- #
    fa = FrameAnalyzer(
        person_segm_weights=cfg.person_segm_weights,
        face_bbox_weights=cfg.face_bbox_weights,
        skin_segm_weights=cfg.skin_segm_weights,
        gender_model_id=cfg.gender_model_id,
        device=cfg.device,
        body_coverage_threshold=cfg.min_frame_person_coverage,
        min_visible_face_threshold=(100.0 - cfg.max_face_mask_percentage),
        gender=cfg.gender_filter,
        debug=cfg.debug,
        enable_body=cfg.enable_body_detection,
        enable_skin=cfg.enable_skin_detection,
        min_skin_pct_threshold=cfg.min_person_skin_percentage,
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
        max_gap=cfg.max_gap,
        num_workers=cfg.num_workers,
    )

    title_gen_factory = _build_title_generator(cfg)
    title_gen = title_gen_factory() if callable(title_gen_factory) else None

    for video in videos_to_process:
        rel_label = video.relative_to(input_path) if input_path.is_dir() else video.name
        if cfg.debug:
            print(f"\n--- Traitement de la vidéo : {rel_label} ---")
        # Construction du chemin de sortie
        if input_path.is_dir():
            out_subdir = Path(cfg.output_dir) / video.relative_to(input_path).parent / video.stem
        else:
            out_subdir = Path(cfg.output_dir) / video.stem

        # Vérification AVANT toute analyse
        try:
            existing_clips = list(out_subdir.glob("*.mp4"))
        except Exception as e:
            existing_clips = []
        if existing_clips:
            print(f"[INFO] Des clips existent déjà pour {rel_label}.")
            resp = input("Voulez-vous retraiter cette vidéo ? (o/n) : ").strip().lower()
            if resp != "o":
                print(f"[INFO] Vidéo {rel_label} ignorée.")
                continue
            # Supprimer les anciens clips
            for clip in existing_clips:
                try:
                    clip.unlink()
                except Exception as e:
                    print(f"[WARN] Impossible de supprimer {clip}: {e}")

        # Tentative de création du dossier de sortie, gestion des noms invalides
        try:
            out_subdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] Impossible de créer le dossier de sortie {out_subdir}: {e}")
            # Proposer un nom nettoyé
            cleaned_stem = "".join(c for c in video.stem if c.isalnum() or c in (" ", "_", "-")).strip()
            cleaned_stem = cleaned_stem.rstrip(". ")  # Supprime points/espaces finaux
            if not cleaned_stem:
                cleaned_stem = "video"
            cleaned_subdir = out_subdir.parent / cleaned_stem
            print(f"[INFO] Proposition de nom corrigé : {cleaned_subdir}")
            resp = input("Utiliser ce nom corrigé ? (o/n) : ").strip().lower()
            if resp == "o":
                try:
                    cleaned_subdir.mkdir(parents=True, exist_ok=True)
                    out_subdir = cleaned_subdir
                except Exception as e2:
                    print(f"[ERROR] Impossible de créer le dossier corrigé : {e2}")
                    print(f"[INFO] Vidéo {rel_label} ignorée.")
                    continue
            else:
                print(f"[INFO] Vidéo {rel_label} ignorée.")
                continue

        # Bloc principal d'analyse et de découpe
        t0 = time.time()
        try:
            clips = va.process(str(video), out_dir=str(out_subdir))
            prefix_base = video.stem
            prefix = "".join(c for c in prefix_base if c.isalnum() or c in (" ", "_", "-")).strip()
            if prefix == "":
                prefix = "video"

            clip_paths = []
            for i, (start, end) in enumerate(clips, 1):
                out_path = Path(out_subdir) / f"{prefix}_edited_{i:03d}.mp4"
                cmd = [
                    "ffmpeg", "-y", "-i", str(video),
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-c", "copy", str(out_path)
                ]
                try:
                    subprocess.run(cmd, check=True)
                    clip_paths.append(out_path)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] ffmpeg failed for clip {out_path.name} of video {video.name}: {e}")
                    continue  # Continue with next clip

            if title_gen:
                if cfg.debug:
                    print("Génération des titres…")
                for p in (tqdm(clip_paths, desc="Titres", leave=False) if cfg.debug else clip_paths):
                    try:
                        title = title_gen.generate_title_from_video(str(p))
                        if not title or title.strip() == "" or title.lower().startswith("clip_"):
                            if cfg.debug:
                                print(f"[TitleGen] Aucun titre généré pour {p.name}, nom conservé.")
                            continue
                        new_path = _rename_clip(p, title)
                        if cfg.debug:
                            print(f"[DEBUG] {p.name} -> {new_path.name}")
                    except Exception as exc:
                        print(f"[TitleGen] Erreur sur {p.name}: {exc}")
            if cfg.debug:
                print("Clips finaux :")
                for p in Path(out_subdir).iterdir():
                    if p.suffix.lower() == ".mp4":
                        print(f"  • {p.name}")
                print(f"--- Fin traitement vidéo : {rel_label} ---\n")

        except Exception as e:
            print(f"[ERROR] Erreur durant le traitement de la vidéo {video}: {e}")
            if cfg.debug:
                print(f"--- Fin traitement vidéo : {rel_label} ---\n")
            continue  # Continue with next video

        elapsed = time.time() - t0

        if cfg.debug:
            print(f"Analyse terminée : {elapsed:.2f}s")
            print(f"Clips créés      : {len(clips)}")
            if clips:
                for i, (s, e) in enumerate(clips, 1):
                    print(f"[DEBUG] Segment {i} : {s:.2f}s -> {e:.2f}s (durée {(e - s):.2f}s)")
            else:
                print("Aucun segment valide.")
        if not clips:
            if cfg.debug:
                print(f"--- Fin traitement vidéo : {rel_label} ---\n")
            continue

    if cfg.debug:
        print("=== AutoCutVideo Fin ===")

if __name__ == "__main__":
    main()
