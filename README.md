# AutoCutVideo

AutoCutVideo fournit un workflow modulaire et performant pour analyser des vidéos, appliquer des critères (NSFW, présence femme, visibilité du visage, pose), découper automatiquement des clips valides et générer des métadonnées via un LLM local (LM Studio). Cible principale Windows 11 + CUDA, fallback Apple Silicon.

---

## Sommaire
- Présentation
- Installation
- Configuration (config.yml)
- CLI et usages
- Spécifications détaillées
- Architecture
- Tests & Qualité
- Performances
- Contribution

---

## Présentation
- Normalisation (future étape): scale+pad vers une résolution/fps cible (par défaut 1280x720@24) avant l'analyse/découpe.
- Détection rapide des cuts (future étape): ffmpeg scene → bornes brutes, refine pour nettoyer les transitions.
- Analyse: détection visage, NSFW, genre (actuellement sur l'image; ROI visage à venir), pose tête; tolérance aux faux négatifs.
- Découpe: segments validés uniquement (stream-copy) selon les critères.
- Description LLM (future étape): X images/clip → JSON par image → vote → {clip}.json.

---

## Installation
- Python 3.10 recommandé.
- Installez via Poetry:
  - poetry install
  - poetry run autocut --help
- ffmpeg requis (concat/cut/probe). Assurez-vous que `ffmpeg` est dans le PATH.

---

## Configuration (config.yml)
Clés principales:
- Entrées/sorties
  - input_video: fichier unique ou dossier
  - output_dir: dossier de sortie
- Exécution
  - device: cuda:0 | cpu
  - num_workers: nombre de threads d'analyse
- Échantillonnage & segments
  - sample_rate: fps d’échantillonnage
  - refine_rate: fps pour l’affinage (réservé)
  - min_clip_duration: durée min d’un clip (s)
  - max_gap: tolérance de trous négatifs en secondes (ex. 10)
- Critères
  - gender_filter: male|female|tous
  - min_face_confidence: seuil détection visage
  - max_face_mask_percentage: % max masqué (min visible = 100 - ce seuil)
  - min_face_bbox_area_pct: % min aire bbox visage vs image
  - min_gender_confidence: confiance min genre
  - max_head_pitch|yaw|roll: seuils pose (deg)
  - enable_*: face/gender/nsfw
  - nsfw_mode: high|medium|low (politique fine au niveau pipeline futur)
- Modèles
  - face_bbox_weights
  - gender_model_id
- Titrage (optionnel)
  - title_generation: enabled, model, prompt, endpoint

Un exemple à jour est présent dans `config.yml`.

---

## CLI et usages
- Analyse + découpe principale
  - poetry run autocut -c config.yml [--debug] [--max-gap-sec S]
  - Génère des clips `*_edited_###.mp4` dans `output_dir/<video_stem>/`.

- Assemblage de clips
  - python -m cli.assemble SRC_DIR DEST_DIR [--min-duration S] [--criteria key:value]
  - Concatène les clips par préfixe (ex. prefix_edited_001.mp4, 002, …) en `prefix_joined.mp4`.
  - Options:
    - --min-duration: filtre de durée minimale (s)
    - --criteria: filtre par métadonnées JSON associées (dot-path support), ex:
      - --criteria describe.voted.nsfw:true
      - --criteria age:old
    - Répéter l’option pour cumuler les critères.

- Test image unique
  - python -m cli.test_frame IMG --config config.yml [--branch full|head] [--debug]

D’autres CLIs (normalize, scenes, describe) seront ajoutées ultérieurement.

---

## Spécifications détaillées (résumé)
- Normalisation (à venir): 1280x720@24 par défaut, letterbox, codec matériel (NVENC/VideoToolbox).
- Cuts (à venir): ffmpeg scene threshold, refine pré/post transitions (fondu).
- Critères (analyse courante):
  - NSFW: wrapper stable, décision finale au niveau pipeline.
  - Présence femme: garder si ≥1 femme (seule ou mixte), rejeter “homme seul/aucune personne”.
  - Visage visible %: sur le genre ciblé par YAML (min visible = 100 - max_face_mask_percentage).
  - Pose: pitch/yaw/roll <= seuils.
  - Tolérance: max_gap (secondes) remplace l’ancien paramètre par frames.
- Description LLM (à venir): prompt chargé depuis prompt.txt, JSON par image (bools, listes, description), vote, résumé, {clip}.json.

---

## Architecture
- cli/
  - process_video.py: orchestration analyse+cut.
  - assemble.py: concat par préfixe avec filtres JSON.
  - test_frame.py: test image/branche.
- pipeline/
  - analyzer.py: FrameAnalyzer (YOLO face, NSFW, genre, pose), VideoAnalyzer (échantillonnage, tolérance, fusion segments).
- detectors/
  - nsfw.py, pose.py, mask.py (visible face).
- config.py: chargement+validation YAML, options titrage.

Modules à venir: normalize, scenes, describe, tracking, criteria, title/metadata, cache.

---

## Tests & Qualité
- Tests unitaires (à écrire dans `tests/`):
  - tests/test_config.py: validation YAML, tolérance, chemins.
  - tests/test_analyzer.py: faux positifs/negatifs, tolérance, pose/visage/genre.
  - tests/test_assemble_utils.py: tri, critères JSON, concat ffmpeg (dry-run/mock).
- Lint/format: flake8, black. CI GitHub Actions recommandé.

---

## Performances
- Favoriser l’early-exit: détecter vite un visage non conforme/femme/NSFW.
- Batching Ultralytics, décodage séquentiel (PyAV/Decord – futur), downscale d’analyse.
- Refine localisé autour des frontières retenues (futur).

---

## Contribution
- Fork → branche feature → PR.
- Respecter PEP8, ajouter des tests.
- Débat ouvert sur les seuils, formats JSON et étapes du workflow.
