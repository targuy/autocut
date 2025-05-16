# config.py
import yaml
from dataclasses import dataclass, field
from typing import Literal, Optional


# --------------------------------------------------------------------------- #
# 1) Sous-structure dédiée à la génération de titres via LMStudio
# --------------------------------------------------------------------------- #
@dataclass
class TitleGenerationConfig:
    """Paramètres facultatifs pour la génération automatisée des titres."""
    enabled: bool = False           # True => on (défaut off)
    prompt: str = ""                # Prompt texte envoyé au modèle
    model: str = ""                 # ID du modèle multimodal (ex. mistral-community_pixtral-12b)
    endpoint: str = "http://localhost:1234/v1/chat/completions"  # URL OpenAI-like


# --------------------------------------------------------------------------- #
# 2) Configuration principale du projet
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- Chemins I/O ---
    input_video: str
    output_dir: str

    # --- Matériel & workers ---
    device: str
    num_workers: int

    # --- Fréquences & découpe ---
    sample_rate: float
    refine_rate: float
    max_gap: float
    min_clip_duration: float

    # --- Seuils & filtres ---
    gender_filter: Literal["homme", "femme", "tous"]
    max_face_mask_percentage: float
    min_skin_percentage: float
    min_gender_confidence: float
    min_face_confidence: float
    max_head_pitch: float
    max_head_yaw: float
    max_head_roll: float

    # --- Poids / modèles ---
    person_segm_weights: str
    face_bbox_weights: str
    skin_segm_weights: str
    gender_model_id: str

    # --- Activation modules ---
    enable_body_detection: bool
    enable_skin_detection: bool
    enable_face_detection: bool
    enable_gender_detection: bool
    enable_nsfw: bool
    nsfw_mode: str

    # --- Debug ---
    debug: bool = False

    # --- Génération de titres ---
    title_generation: TitleGenerationConfig = field(
        default_factory=TitleGenerationConfig
    )


# --------------------------------------------------------------------------- #
# 3) Loader YAML -> Config
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Config:
    """
    Charge le fichier YAML et retourne un objet Config entièrement typé.
    Les valeurs manquantes sont renseignées avec les valeurs par défaut.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Sous-section facultative "title_generation":
    tg_raw = raw.get("title_generation", {}) or {}

    return Config(
        # --- Chemins I/O ---
        input_video=raw["input_video"],
        output_dir=raw["output_dir"],

        # --- Matériel & workers ---
        device=raw.get("device", "cuda:0"),
        num_workers=raw.get("num_workers", 4),

        # --- Fréquences & découpe ---
        sample_rate=raw.get("sample_rate", 1.0),
        refine_rate=raw.get("refine_rate", 24.0),
        max_gap=raw.get("max_gap", 0.2),
        min_clip_duration=raw.get("min_clip_duration", 5.0),

        # --- Seuils & filtres ---
        gender_filter=raw.get("gender_filter", "tous"),
        max_face_mask_percentage=raw.get("max_face_mask_percentage", 25.0),
        min_skin_percentage=raw.get("min_skin_percentage", 50.0),
        min_gender_confidence=raw.get("min_gender_confidence", 0.8),
        min_face_confidence=raw.get("min_face_confidence", 0.25),
        max_head_pitch=raw.get("max_head_pitch", 20.0),
        max_head_yaw=raw.get("max_head_yaw", 30.0),
        max_head_roll=raw.get("max_head_roll", 20.0),

        # --- Poids / modèles ---
        person_segm_weights=raw["person_segm_weights"],
        face_bbox_weights=raw["face_bbox_weights"],
        skin_segm_weights=raw["skin_segm_weights"],
        gender_model_id=raw["gender_model_id"],

        # --- Activation modules ---
        enable_body_detection=raw.get("enable_body_detection", True),
        enable_skin_detection=raw.get("enable_skin_detection", True),
        enable_face_detection=raw.get("enable_face_detection", True),
        enable_gender_detection=raw.get("enable_gender_detection", True),
        enable_nsfw=raw.get("enable_nsfw", True),
        nsfw_mode=raw.get("nsfw_mode", "high"),

        # --- Debug ---
        debug=raw.get("debug", False),

        # --- Génération de titres ---
        title_generation=TitleGenerationConfig(
            enabled=tg_raw.get("enabled", False),
            prompt=tg_raw.get("prompt", ""),
            model=tg_raw.get("model", ""),
            endpoint=tg_raw.get("endpoint", "http://localhost:1234/v1/chat/completions")
        ),
    )
