import yaml
from dataclasses import dataclass
from typing import Literal

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
    gender_filter: Literal["homme", "femme", "tous"]
    max_face_mask_percentage: float
    min_skin_percentage: float

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

    min_gender_confidence: float
    min_face_confidence: float

    max_head_pitch: float
    max_head_yaw: float
    max_head_roll: float

    debug: bool = False

def load_config(path: str) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return Config(
        input_video               = raw["input_video"],
        output_dir                = raw["output_dir"],
        device                    = raw.get("device", "cuda:0"),
        num_workers               = raw.get("num_workers", 4),
        sample_rate               = raw.get("sample_rate", 1.0),
        refine_rate               = raw.get("refine_rate", 24.0),
        max_gap                   = raw.get("max_gap", 0.2),
        min_clip_duration         = raw.get("min_clip_duration", 5.0),
        gender_filter             = raw.get("gender_filter", "tous"),
        max_face_mask_percentage  = raw.get("max_face_mask_percentage", 25.0),
        min_skin_percentage       = raw.get("min_skin_percentage", 50.0),
        person_segm_weights       = raw["person_segm_weights"],
        face_bbox_weights         = raw["face_bbox_weights"],
        skin_segm_weights         = raw["skin_segm_weights"],
        gender_model_id           = raw["gender_model_id"],
        enable_body_detection     = raw.get("enable_body_detection", True),
        enable_skin_detection     = raw.get("enable_skin_detection", True),
        enable_face_detection     = raw.get("enable_face_detection", True),
        enable_gender_detection   = raw.get("enable_gender_detection", True),
        enable_nsfw               = raw.get("enable_nsfw", True),
        nsfw_mode                 = raw.get("nsfw_mode", "high"),
        min_gender_confidence     = raw.get("min_gender_confidence", 0.8),
        min_face_confidence       = raw.get("min_face_confidence", 0.25),
        max_head_pitch            = raw.get("max_head_pitch", 20.0),
        max_head_yaw              = raw.get("max_head_yaw", 30.0),
        max_head_roll             = raw.get("max_head_roll", 20.0),
        debug                     = raw.get("debug", False),
    )
