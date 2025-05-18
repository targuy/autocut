# config.py
import sys
import os
from pathlib import Path
import yaml
from dataclasses import dataclass, field
from typing import Literal

# -----------------------------------------------------------------------------
# 1) Sous-structure dédiée à la génération de titres via LMStudio
# -----------------------------------------------------------------------------
@dataclass
class TitleGenerationConfig:
    """Paramètres facultatifs pour la génération automatisée des titres."""
    enabled: bool = False      # True => activation du titrage
    prompt: str = ""
    model: str = ""
    endpoint: str = "http://localhost:1234/v1/chat/completions"


# -----------------------------------------------------------------------------
# 2) Configuration principale du projet
# -----------------------------------------------------------------------------
@dataclass
class Config:
    input_video: str
    output_dir: str

    device: str
    num_workers: int

    sample_rate: float
    refine_rate: float
    max_gap: float
    min_clip_duration: float

    gender_filter: Literal["homme", "femme", "tous", "male", "female", "all"]
    max_face_mask_percentage: float
    min_skin_percentage: float
    min_gender_confidence: float
    min_face_confidence: float
    max_head_pitch: float
    max_head_yaw: float
    max_head_roll: float

    person_segm_weights: str
    face_bbox_weights: str
    skin_segm_weights: str
    gender_model_id: str

    enable_body_detection: bool
    enable_skin_detection: bool
    enable_face_detection: bool
    enable_gender_detection: bool
    enable_nsfw: bool
    nsfw_mode: str

    debug: bool = False

    # Génération de titres par LMStudio
    title_generation: TitleGenerationConfig = field(default_factory=TitleGenerationConfig)


# -----------------------------------------------------------------------------
# 3) Loader YAML -> Config, avec validation et normalisation des chemins
# -----------------------------------------------------------------------------
def load_config(path: str) -> Config:
    # 3.1) Chargement et parsing YAML
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Fichier de config introuvable : {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Erreur de parsing YAML ({path}) :\n{e}")
        sys.exit(1)

    # 3.2) Vérification des clés obligatoires
    required = [
        "input_video", "output_dir",
        "person_segm_weights", "face_bbox_weights",
        "skin_segm_weights", "gender_model_id"
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        print(f"[ERROR] Paramètres manquants dans config.yaml : {missing}")
        sys.exit(1)

    # 3.3) Validation des types et cohérence des paramètres
    # Normaliser les chemins d'entrée et de sortie pour vérifications
    def norm(p: str) -> str:
        """
        Nettoie et normalise un chemin :
        - Enlève les quotes
        - Développe ~
        - Uniformise les slashes
        """
        if not isinstance(p, str):
            raise ValueError(f"Le chemin spécifié n’est pas une chaîne de caractères valide : {p}")
        p = p.strip().strip('"').strip("'")  # Enlève les guillemets simples ou doubles
        return os.path.normpath(os.path.expanduser(p))

    # Normaliser les chemins
    try:
        input_path = norm(raw["input_video"])
        output_path = norm(raw["output_dir"])
    except KeyError as e:
        print(f"[ERREUR] Clé manquante dans config.yaml : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERREUR] Problème avec les chemins fournis : {e}")
        sys.exit(1)

    # Vidéo d'entrée : doit exister et être un fichier lisible
    if not os.path.isfile(input_path):
        print(f"[ERROR] Vidéo d'entrée introuvable ou illisible : {raw['input_video']}")
        sys.exit(1)
    # Dossier de sortie : ne doit pas pointer vers un fichier
    if os.path.isfile(output_path):
        print(f"[ERROR] Le chemin de sortie spécifié correspond à un fichier (dossier attendu) : {raw['output_dir']}")
        sys.exit(1)
    # Vérifier l'existence des fichiers de poids requis pour YOLO
    person_weights_path = norm(raw["person_segm_weights"])
    if not os.path.isfile(person_weights_path):
        print(f"[ERROR] Fichier de poids introuvable : {raw['person_segm_weights']}")
        sys.exit(1)
    face_weights_path = norm(raw["face_bbox_weights"])
    if not os.path.isfile(face_weights_path):
        print(f"[ERROR] Fichier de poids introuvable : {raw['face_bbox_weights']}")
        sys.exit(1)
    skin_weights_path = norm(raw["skin_segm_weights"])
    if not os.path.isfile(skin_weights_path):
        print(f"[ERROR] Fichier de poids introuvable : {raw['skin_segm_weights']}")
        sys.exit(1)
    # Champ device : peut être un index GPU entier ou une chaîne ("cuda:0", "cpu")
    if "device" in raw:
        dev_val = raw["device"]
        if isinstance(dev_val, int):
            if dev_val >= 0:
                raw["device"] = f"cuda:{dev_val}"
            else:
                print(f"[ERROR] Valeur invalide pour 'device' : {dev_val}")
                sys.exit(1)
        elif not isinstance(dev_val, str):
            print(f"[ERROR] Type invalide pour 'device' (chaîne attendue, obtenu {type(dev_val).__name__})")
            sys.exit(1)
    # Champs numériques (float)
    numeric_fields = ["sample_rate", "refine_rate", "max_gap", "min_clip_duration",
                      "max_face_mask_percentage", "min_skin_percentage",
                      "min_gender_confidence", "min_face_confidence",
                      "max_head_pitch", "max_head_yaw", "max_head_roll"]
    for field in numeric_fields:
        if field in raw:
            val = raw[field]
            if type(val) not in (int, float):
                print(f"[ERROR] Type invalide pour '{field}' (nombre attendu, obtenu {type(val).__name__})")
                sys.exit(1)
    # Champs entiers (int)
    if "num_workers" in raw:
        val = raw["num_workers"]
        if type(val) not in (int,):
            if isinstance(val, float) and val.is_integer():
                raw["num_workers"] = int(val)
            else:
                print(f"[ERROR] Type invalide pour 'num_workers' (entier attendu, obtenu {type(val).__name__})")
                sys.exit(1)
    # Champs booléens
    bool_fields = ["enable_body_detection", "enable_skin_detection", "enable_face_detection",
                   "enable_gender_detection", "enable_nsfw", "debug"]
    for field in bool_fields:
        if field in raw and not isinstance(raw[field], bool):
            print(f"[ERROR] Type invalide pour '{field}' (booléen attendu, obtenu {type(raw[field]).__name__})")
            sys.exit(1)
    # Champs string
    str_fields = ["nsfw_mode", "gender_model_id"]
    for field in str_fields:
        if field in raw and not isinstance(raw[field], str):
            print(f"[ERROR] Type invalide pour '{field}' (chaîne attendue, obtenu {type(raw[field]).__name__})")
            sys.exit(1)
    # Validation du filtre de genre (homme/femme/tous ou male/female/all)
    if "gender_filter" in raw:
        if not isinstance(raw["gender_filter"], str):
            print(f"[ERROR] Type invalide pour 'gender_filter' (chaîne attendue, obtenu {type(raw['gender_filter']).__name__})")
            sys.exit(1)
        gf_val = raw["gender_filter"].lower()
        if gf_val in ("homme", "male", "man"):
            gf_val = "male"
        elif gf_val in ("femme", "female", "woman"):
            gf_val = "female"
        elif gf_val in ("tous", "all"):
            gf_val = "tous"
        else:
            print(f"[ERROR] Valeur invalide pour 'gender_filter' : {raw['gender_filter']}")
            sys.exit(1)
        raw["gender_filter"] = gf_val
    # Validation de la section title_generation si activée
    if raw.get("title_generation", {}) and raw["title_generation"].get("enabled", False):
        tg = raw["title_generation"]
        if not isinstance(tg.get("enabled"), bool):
            print(f"[ERROR] Type invalide pour 'title_generation.enabled' (booléen attendu)")
            sys.exit(1)
        if not tg.get("model"):
            print("[ERROR] 'title_generation' est activé mais aucun modèle n'est spécifié dans 'model'.")
            sys.exit(1)
        if "model" in tg and not isinstance(tg["model"], str):
            print(f"[ERROR] Type invalide pour 'title_generation.model' (chaîne attendue)")
            sys.exit(1)
        if "prompt" in tg and not isinstance(tg.get("prompt", ""), str):
            print(f"[ERROR] Type invalide pour 'title_generation.prompt' (chaîne attendue)")
            sys.exit(1)
        if "endpoint" in tg and not isinstance(tg.get("endpoint", ""), str):
            print(f"[ERROR] Type invalide pour 'title_generation.endpoint' (chaîne attendue)")
            sys.exit(1)

    # 3.4) Construction du Config (avec normalisation des chemins principaux)
    return Config(
        input_video               = norm(raw["input_video"]),
        output_dir                = norm(raw["output_dir"]),

        device                    = raw.get("device", "cuda:0"),
        num_workers               = raw.get("num_workers", 4),

        sample_rate               = raw.get("sample_rate", 1.0),
        refine_rate               = raw.get("refine_rate", 24.0),
        max_gap                   = raw.get("max_gap", 0.2),
        min_clip_duration         = raw.get("min_clip_duration", 5.0),

        gender_filter             = raw.get("gender_filter", "tous"),
        max_face_mask_percentage  = raw.get("max_face_mask_percentage", 25.0),
        min_skin_percentage       = raw.get("min_skin_percentage", 50.0),
        min_gender_confidence     = raw.get("min_gender_confidence", 0.8),
        min_face_confidence       = raw.get("min_face_confidence", 0.25),
        max_head_pitch            = raw.get("max_head_pitch", 20.0),
        max_head_yaw              = raw.get("max_head_yaw", 30.0),
        max_head_roll             = raw.get("max_head_roll", 20.0),

        person_segm_weights       = norm(raw["person_segm_weights"]),
        face_bbox_weights         = norm(raw["face_bbox_weights"]),
        skin_segm_weights         = norm(raw["skin_segm_weights"]),
        gender_model_id           = raw["gender_model_id"],

        enable_body_detection     = raw.get("enable_body_detection", True),
        enable_skin_detection     = raw.get("enable_skin_detection", True),
        enable_face_detection     = raw.get("enable_face_detection", True),
        enable_gender_detection   = raw.get("enable_gender_detection", True),
        enable_nsfw               = raw.get("enable_nsfw", True),
        nsfw_mode                 = raw.get("nsfw_mode", "high"),

        debug                     = raw.get("debug", False),

        title_generation = TitleGenerationConfig(
            enabled  = raw.get("title_generation", {}).get("enabled", False),
            prompt   = raw.get("title_generation", {}).get("prompt", ""),
            model    = raw.get("title_generation", {}).get("model", ""),
            endpoint = raw.get("title_generation", {}).get("endpoint", TitleGenerationConfig.endpoint)
        )
    )
