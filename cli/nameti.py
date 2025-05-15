import requests
import base64
import mimetypes
import os

# Chemin vers l'image (remplacez par votre chemin réel)
image_path = "T:/IMG src/testImg/e6a651ef103024b7ba4a9d5c9e9ec016.28.jpg"

# Vérifier si le fichier image existe
if not os.path.exists(image_path):
    print(f"Erreur : Le fichier image n'existe pas à {image_path}")
    exit(1)

# Déterminer le type MIME de l'image
mime_type, _ = mimetypes.guess_type(image_path)
if mime_type is None:
    mime_type = "application/octet-stream"  # Type par défaut

# Lire et encoder l'image en base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Construire le message avec le prompt
prompt = "describe the picture of a woman giving that criterias only: ethnicity,is she nude or wearing lingerie,hair color, age, body shape, breasts size,breasts shape, sexual activity type,sex pose name. concatenate answers without the criteria name using _ as a separator answers to make a video file title"
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
    ]
}

# Construire la charge utile (payload) pour l'API
payload = {
    "model": "mistral-community_pixtral-12b",
    "messages": [message]
}

# URL de l'endpoint API (ajustez le port si nécessaire)
endpoint = "http://localhost:1234/v1/chat/completions"

# Envoyer la requête
try:
    response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()  # Lever une exception pour les codes d'erreur HTTP
    result = response.json()
    description = result["choices"][0]["message"]["content"]
    print("Titre du fichier vidéo généré :", description)
except requests.exceptions.RequestException as e:
    print("Erreur lors de l'envoi de la requête :", e)
except KeyError:
    print("Erreur : Format de réponse inattendu")