import argparse
import base64
import mimetypes
import os
import subprocess
import tempfile
import requests

# Extensions de fichiers acceptées
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# Nombre maximum de tentatives pour reformuler la réponse
MAX_RETRIES = 4

# Mots-clés indiquant un échec de compréhension (tous en minuscules)
BAD_KEYWORDS = [
    "color", "age", "body", "shape", "breasts", "size",
    "breasts", "shape", "sexual", "activity", "type", "sex", "pose", "name"
]

def process_image(image_path):
    """Traite une image : encode, envoie à l'API, et retente si la réponse contient encore un terme invalide."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "describe the picture of a woman giving that criterias only: "
        "ethnicity,is she nude or wearing lingerie,hair color, age, "
        "body shape, breasts size,breasts shape, sexual activity type,sex pose name. "
        "concatenate answers without the criteria name using _ as a separator answers to make a video file title"
    )

    payload = {
        "model": "mistral-community_pixtral-12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                ]
            }
        ]
    }

    endpoint = "http://localhost:1234/v1/chat/completions"
    last_response = None

    # Boucle de retry
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        last_response = content

        # Vérifier si un des mots interdits est présent
        lower = content.lower()
        if not any(keyword in lower for keyword in BAD_KEYWORDS):
            return content  # Compréhension OK

    # Si on atteint ici, on renvoie la dernière réponse malgré tout
    return last_response


def process_video(video_path):
    """Extrait l'image médiane d'une vidéo, la traite et renvoie la description."""
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())
    middle_time = duration / 2

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)

    try:
        extract_cmd = [
            "ffmpeg", "-y", "-ss", str(middle_time), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", "-update", "1", tmp_path
        ]
        subprocess.run(
            extract_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        description = process_image(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return description


def sanitize_filename(name):
    """Supprime ou remplace les caractères invalides pour un nom de fichier."""
    return ''.join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()


def rename_file(src_path, new_base):
    """Renomme le fichier src_path avec la base new_base, gère les doublons."""
    directory, old_name = os.path.split(src_path)
    ext = os.path.splitext(old_name)[1]
    base = sanitize_filename(new_base)
    candidate = f"{base}{ext}"
    dest = os.path.join(directory, candidate)
    counter = 1
    while os.path.exists(dest):
        candidate = f"{base}_{counter}{ext}"
        dest = os.path.join(directory, candidate)
        counter += 1
    os.rename(src_path, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Générer un titre à partir d'une image, d'un répertoire ou d'une vidéo."
    )
    parser.add_argument("input_path", help="Chemin vers l'image, le répertoire ou la vidéo.")
    parser.add_argument(
        "input_type",
        choices=["image", "directory", "video"],
        help="Type d'entrée : 'image', 'directory' ou 'video'."
    )
    parser.add_argument(
        "--rename", action="store_true",
        help="Renommer le fichier d'origine avec le nouveau titre"
    )
    args = parser.parse_args()

    def handle(path, func):
        try:
            desc = func(path)
            ext = os.path.splitext(path)[1] or ''
            title_with_ext = f"{desc}{ext}"
            if args.rename:
                new_path = rename_file(path, desc)
                print(f"{path} -> {new_path}")
            else:
                print(f"{path} | {title_with_ext}")
        except Exception as e:
            print(f"Erreur lors du traitement de {path} : {e}")

    if args.input_type == "directory":
        for fname in os.listdir(args.input_path):
            fpath = os.path.join(args.input_path, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                handle(fpath, process_image)
            elif ext in VIDEO_EXTENSIONS:
                handle(fpath, process_video)
    else:
        if args.input_type == "image":
            handle(args.input_path, process_image)
        elif args.input_type == "video":
            handle(args.input_path, process_video)

if __name__ == "__main__":
    main()
