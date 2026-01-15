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

AutoCutVideo provides automated installation scripts that handle environment setup, FFmpeg installation, and dependency management with GPU/CPU support detection.

### Prérequis

- **Python 3.10 ou supérieur**
- **FFmpeg** (sera installé automatiquement si manquant)
- Pour le support GPU NVIDIA : CUDA 12.1 ou supérieur et pilotes NVIDIA à jour

### Option 1 : Installation automatique (Recommandée)

#### Linux / macOS

```bash
git clone https://github.com/targuy/autocut.git
cd autocut
chmod +x install.sh
./install.sh
```

Le script d'installation va :
- Vérifier la version de Python
- Détecter et installer FFmpeg si nécessaire
- Créer un environnement virtuel Python
- Vous proposer le choix entre installation CPU ou CUDA
- Installer toutes les dépendances requises

#### Windows

```cmd
git clone https://github.com/targuy/autocut.git
cd autocut
install.bat
```

Le script d'installation va :
- Vérifier la version de Python
- Détecter FFmpeg (propose imageio-ffmpeg si manquant)
- Créer un environnement virtuel Python
- Vous proposer le choix entre installation CPU ou CUDA
- Installer toutes les dépendances requises

### Option 2 : Installation avec Poetry

Si vous préférez utiliser Poetry :

1. **Installer Poetry** (si nécessaire)  
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Installer les dépendances**

   Pour CPU uniquement :
   ```bash
   poetry install --extras cpu
   ```

   Pour GPU NVIDIA (CUDA) :
   ```bash
   poetry install --extras cuda
   ```

   Avec toutes les fonctionnalités :
   ```bash
   poetry install --extras all
   ```

3. **Activer l'environnement**  
   ```bash
   poetry shell
   ```

### Option 3 : Installation manuelle avec pip

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scriptsctivate.bat  # Windows

# Installer pour CPU
pip install -e ".[cpu]"

# OU installer pour CUDA
pip install -e ".[cpu]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```

### Gestion des conflits GPU/CPU

Le projet utilise des **extras Poetry** pour éviter les conflits entre packages CPU et GPU :

- `[cpu]` : Installe `torch`, `torchvision`, et `onnxruntime` (versions CPU)
- `[cuda]` : Installe les versions CPU puis permet l'upgrade manuel vers CUDA
- `[ffmpeg]` : Installe `imageio-ffmpeg` pour télécharger FFmpeg automatiquement
- `[all]` : Installe toutes les dépendances de base

**Note importante** : Les packages `torch` et `onnxruntime` ont des versions séparées pour CPU et GPU. Le script d'installation gère automatiquement ce conflit en installant d'abord les dépendances de base, puis en upgradant vers les versions GPU si sélectionné.

### Vérification de l'installation

```bash
# Vérifier FFmpeg
ffmpeg -version

# Vérifier PyTorch et la détection CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Vérifier ONNX Runtime
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}'); print(f'Providers: {onnxruntime.get_available_providers()}')"
```

---

**📖 Pour plus de détails** : Consultez [INSTALL.md](INSTALL.md) pour un guide d'installation complet avec dépannage et [REQUIREMENTS_ANALYSIS.md](REQUIREMENTS_ANALYSIS.md) pour comprendre la gestion des conflits GPU/CPU.



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

Une fois l'installation terminée, vous avez plusieurs options pour exécuter AutoCutVideo :

### Option 1 : Scripts de lancement automatiques (Recommandé)

#### Linux / macOS
```bash
./run.sh --config config.yaml
```

#### Windows
```cmd
run.bat --config config.yaml
```

Ces scripts activent automatiquement l'environnement virtuel et lancent l'application.

### Option 2 : Avec environnement virtuel activé

```bash
# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate.bat  # Windows

# Lancer l'application
python main.py --config config.yaml
```

### Option 3 : Avec Poetry

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
