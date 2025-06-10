"""
pipeline/title_generator.py
Module pour la génération de titres de clips à l'aide d'un modèle multimodal via LMStudio.
"""
import base64
import requests
import subprocess
import time
from requests.exceptions import RequestException
import cv2
from pathlib import Path

def ensure_server_running(model: str, endpoint: str) -> bool:
    """
    Vérifie que le serveur LMStudio est en cours d'exécution et que le modèle est chargé.
    Tente de lancer le serveur et de charger le modèle via la CLI `lms` si nécessaire.
    Retourne True si le modèle est prêt à l'inférence, False sinon.
    """
    # Extraire l'URL de base (jusqu'à /v1) à partir de l'endpoint fourni
    base_url = endpoint
    if "/chat/completions" in base_url:
        base_url = base_url.split("/chat/completions")[0]
    # S'assurer que base_url se termine par /v1
    if not base_url.endswith("/v1"):
        if "/v1" in base_url:
            base_url = base_url[:base_url.index("/v1")+3]
        else:
            base_url = base_url.rstrip("/") + "/v1"
    # Vérifier si le serveur est joignable
    try:
        resp = requests.get(f"{base_url}/models", timeout=2)
        server_up = resp.status_code == 200
    except RequestException:
        server_up = False
    if not server_up:
        # Tenter de démarrer le serveur LMStudio via `lms server start`
        try:
            subprocess.Popen(
                ["lms", "server", "start"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[TitleGeneration] ERREUR : l'outil `lms` n'est pas disponible pour lancer LMStudio.")
            return False
        # Attendre que le serveur soit accessible
        max_wait = 10  # secondes
        waited = 0
        server_up = False
        while waited < max_wait:
            try:
                resp = requests.get(f"{base_url}/models", timeout=2)
                if resp.status_code == 200:
                    server_up = True
                    break
            except RequestException:
                server_up = False
            time.sleep(1)
            waited += 1
        if not server_up:
            print("[TitleGeneration] ERREUR : impossible de démarrer le serveur LMStudio.")
            return False
    # À ce stade, le serveur est démarré. Vérifier si le modèle est chargé.
    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except RequestException as e:
        print(f"[TitleGeneration] ERREUR : échec de la récupération des modèles chargés ({e}).")
        return False
    loaded_models = []
    if isinstance(data, dict) and "data" in data:
        # Format OpenAI-like
        loaded_models = [m.get("id") for m in data["data"] if m.get("id")]
    elif isinstance(data, list):
        loaded_models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    # Vérifier si le modèle est déjà dans la liste des modèles chargés
    model_identifier = model
    alt_identifier = None
    if "/" in model_identifier:
        alt_identifier = model_identifier.split("/")[-1]
    elif "_" in model_identifier:
        alt_identifier = model_identifier.split("_")[-1]
    model_loaded = False
    for mid in loaded_models:
        if mid == model_identifier or (alt_identifier and mid == alt_identifier):
            model_loaded = True
            break
    if not model_loaded:
        # Tenter de charger le modèle via `lms load`
        model_to_load = model_identifier
        if "_" in model_identifier and "/" not in model_identifier:
            parts = model_identifier.split("_", 1)
            if len(parts) == 2:
                model_to_load = f"{parts[0]}/{parts[1]}"
        try:
            subprocess.run(["lms", "load", model_to_load], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[TitleGeneration] ERREUR : échec du chargement du modèle '{model_identifier}' via lms ({e}).")
            return False
    return True

def generate_titles(image_paths, prompt: str, model: str, endpoint: str):
    """
    Envoie chaque image de la liste au modèle multimodal via l'API OpenAI-compatible de LMStudio,
    et génère une liste de titres (un par image) en nettoyant et dédupliquant les résultats.
    """
    # Fonction interne pour traiter une image individuelle
    def process_image(img_path: str):

        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"[TitleGeneration] Impossible de lire l'image {img_path} : {e}")
            return None
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_data}"}
                    ]
                }
            ],
            "max_tokens": 64
        }
        # Envoyer la requête
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
        except RequestException as e:
            print(f"[TitleGeneration] Requête échouée pour {img_path} : {e}")
            return None
        # Extraire le contenu du titre depuis la réponse
        data = response.json()
        if "choices" not in data or not data["choices"]:
            return None
        content = data["choices"][0]["message"]["content"]
        # S'assurer d'obtenir une chaîne de caractères
        if isinstance(content, list):
            content_str = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            content_str = str(content)
        return content_str

    titles = []
    if image_paths:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(4, len(image_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, img) for img in image_paths]
            results = [f.result() for f in futures]  # préserver l'ordre
    else:
        results = []

    # Nettoyer les titres bruts et gérer les échecs
    for i, res in enumerate(results, start=1):
        if res is None or not res.strip():
            titles.append(f"clip_{i:03d}")
            continue
        title = res.strip()
        # Retirer d'éventuels guillemets englobants
        if (title.startswith('"') and title.endswith('"')) or (title.startswith("'") and title.endswith("'")):
            title = title[1:-1].strip()
        # Retirer un préfixe du style "Title: " ou "Titre : "
        low = title.lower()
        if low.startswith("title:") or low.startswith("titre:"):
            colon_idx = title.find(":")
            if colon_idx != -1:
                title = title[colon_idx+1:].strip()
        # Retirer les caractères non autorisés dans un nom de fichier
        forbidden = '\\/:*?"<>|'
        for ch in forbidden:
            title = title.replace(ch, "")
        # Retirer les espaces ou points en fin de titre
        title = title.rstrip(". ").strip()
        # Utiliser un nom par défaut si le titre est vide après nettoyage
        if title == "":
            titles.append(f"clip_{i:03d}")
        else:
            titles.append(title)
    # Déduplication des titres (ajout d'un suffixe si nécessaire)
    seen = {}
    for idx, name in enumerate(titles):
        if name in seen:
            count = seen[name]
            new_name = f"{name}_{count+1}"
            # Trouver un nom unique non encore utilisé
            while new_name in seen or new_name in titles:
                count += 1
                new_name = f"{name}_{count+1}"
            titles[idx] = new_name
            seen[name] = count + 1
            seen[new_name] = 1
        else:
            seen[name] = 1
    return titles

class TitleGenerator:
    """Générateur de titres basé sur un modèle multimodal via LMStudio."""
    def __init__(self, endpoint: str, model: str, prompt_template: str):
        self.endpoint = endpoint
        self.model = model
        self.prompt_template = prompt_template

    def generate_title(self, image_path: str) -> str:
        """Génère un titre unique à partir d'une image donnée."""
        titles = generate_titles([image_path], self.prompt_template, self.model, self.endpoint)
        # Retourner le premier titre généré (une seule image en entrée)
        return titles[0] if titles else ""

    def generate_title_from_video(self, video_path: str) -> str:
        """Extrait une image médiane de la vidéo et génère un titre."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            mid_frame_idx = frame_count // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        else:
            # Si le nombre de frames est indisponible, utiliser la position relative
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.5)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Impossible d'extraire une image de {video_path}")
        # Sauvegarder la frame dans un fichier temporaire
        video_path_obj = Path(video_path)
        img_path = video_path_obj.with_suffix(".jpg")
        success = cv2.imwrite(str(img_path), frame)
        if not success:
            raise RuntimeError(f"Impossible de sauvegarder l'image {img_path}")
        try:
            # Générer le titre à partir de l'image
            title = self.generate_title(str(img_path))
        finally:
            # Nettoyer le fichier image temporaire
            try:
                img_path.unlink()
            except OSError:
                pass
        return title
