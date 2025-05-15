# AutoCutVideo

AutoCutVideo est un pipeline Python modulaire et performant pour analyser des vidéos, détecter des personnes, classifier leur genre (homme/femme), mesurer l’exposition du visage et la proportion de peau visible, puis découper automatiquement des clips répondant à des critères configurables.

---

## Table des matières

- [Fonctionnalités](#fonctionnalités)  
- [Architecture et modules](#architecture-et-modules)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage CLI principal](#usage-cli-principal)  
- [Tests CLI sur image unique](#tests-cli-sur-image-unique)  
- [Exemple d’exécution](#exemple-dexécution)  
- [Développement & tests](#développement--tests)  
- [Contribuer](#contribuer)  
- [Licence](#licence)  

---

## Fonctionnalités

- **Détection de personnes** via YOLOv8 segmentation (`person_yolov8m-seg`).  
- **Localisation du visage** via YOLOv8 détection (`face_yolov8m.pt`).  
- **Mesure du masquage du visage** (pourcentage masqué).  
- **Segmentation de la peau** via YOLOv8 segmentation (`skin_yolov8m-seg`).  
- **Classification du genre** (“homme” / “femme” / “tous”) via un modèle Hugging Face Transformers local.  
- **Échantillonnage** à 1 fps (configurable) pour vitesse, puis **affinage** à 24 fps pour trouver l’entrée/sortie exactes.  
- **Découpe** de segments continus satisfaisant les critères (durée minimale configurable) en clips MP4 sans réencodage.

---

## Architecture et modules

```
AutoCutVideo/
├── cli/
│   ├── process_video.py        # Point d’entrée principal CLI
│   ├── test_person.py          # CLI test segmentation corps
│   ├── test_face.py            # CLI test détection visage
│   ├── test_skin.py            # CLI test segmentation peau
│   └── test_gender.py          # CLI test classification genre
├── config.py                   # Loader YAML → dataclass Config
├── config.yaml                 # Paramètres par défaut
├── detectors/
│   └── mask.py                 # Calcul % visage masqué
├── segmenters/
│   └── skin.py                 # Segmentation peau
├── pipeline/
│   └── analyzer.py             # FrameAnalyzer & VideoAnalyzer
├── utils.py                    # Fonctions utilitaires
├── tests/                      # Tests pytest
└── README.md                   # Cette documentation
```

---

## Installation

1. **Cloner le dépôt**  
   ```bash
   git clone https://votre-repo/AutoCutVideo.git
   cd AutoCutVideo
   ```

2. **Installer Poetry** (si nécessaire)  
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Installer les dépendances**  
   ```bash
   poetry install
   ```

4. **Activer l’environnement**  
   ```bash
   poetry shell
   ```

---

## Configuration

Les paramètres sont centralisés dans `config.yaml`. Exemple :

```yaml
input_video:                "E:/Videos/input.mp4"
output_dir:                 "E:/Videos/clips"
device:                     "cuda"
num_workers:                4
sample_rate:                1.0
refine_rate:                24.0
min_clip_duration:          5.0
gender_filter:              "tous"
max_face_mask_percentage:   25.0
min_skin_percentage:        50.0
person_segm_weights:        "E:/.../person_yolov8m-seg.pt"
face_bbox_weights:          "E:/.../face_yolov8m.pt"
skin_segm_weights:          "E:/.../skin_yolov8m-seg_400.pt"
gender_model_dir:           "E:/.../gender/rizandwiki-gender"
```

---

## Usage CLI principal

Une fois configuré, lancez :

```bash
python main.py --config config.yaml
```

Ou, si vous avez défini le script Poetry :

```bash
poetry run autocut --config config.yaml
```

Les clips seront générés dans le dossier `output_dir` spécifié. La console affichera le nombre de clips créés et leurs intervalles.

---

## Tests CLI sur image unique

Pour tester chaque composant indépendamment sur une seule image :

- **test_person.py** : segmentation du corps  
  ```bash
  python cli/test_person.py chemin/vers/image.jpg --config config.yaml
  ```
- **test_face.py** : détection de visages  
  ```bash
  python cli/test_face.py chemin/vers/image.jpg --config config.yaml
  ```
- **test_skin.py** : segmentation de la peau et pourcentage visible  
  ```bash
  python cli/test_skin.py chemin/vers/image.jpg --config config.yaml
  ```
- **test_gender.py** : classification du genre  
  ```bash
  python cli/test_gender.py chemin/vers/image.jpg --config config.yaml
  ```

Chaque script renvoie du JSON avec les résultats (BBOX, pourcentages, labels, confiances…).

---

## Exemple d’exécution

```bash
$ python main.py -c config.yaml
[INFO] Chargement de la configuration…
[INFO] Initialisation des modèles sur cuda
[INFO] Vidéo ouverte (fps=24.00, durée=1200.0s)
[INFO] Échantillonnage à 1.0 fps, affinage à 24.0 fps
[✔] Généré 12 clip(s) dans « E:/Videos/clips »
```

---

## Développement & tests

- **Tests unitaires** :  
  ```bash
  pytest --maxfail=1 --disable-warnings -q
  ```
- **Lint & format** :  
  ```bash
  flake8 .
  black .
  ```
- **CI/CD** : GitHub Actions intégré pour tests, lint, coverage.

---

## Contribuer

1. Forkez le projet.  
2. Créez une branche (`feature/nom`).  
3. Ajoutez votre code et tests.  
4. Ouvrez un pull request.  

Merci de respecter le style PEP 8 et d’ajouter des tests pour toute nouvelle fonctionnalité.

---

## Licence

MIT License © 2025 Benoit Guitard. Voir le fichier `LICENSE` pour plus de détails.
