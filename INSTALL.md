# Guide d'Installation AutoCutVideo

Ce guide détaillé couvre l'installation d'AutoCutVideo sur différentes plateformes et la résolution des problèmes courants.

## Table des matières

- [Prérequis](#prérequis)
- [Installation automatique](#installation-automatique)
- [Installation manuelle](#installation-manuelle)
- [Résolution des conflits GPU/CPU](#résolution-des-conflits-gpucpu)
- [Dépannage](#dépannage)
- [Vérification de l'installation](#vérification-de-linstallation)

## Prérequis

### Systèmes supportés

- **Linux** : Ubuntu 20.04+, Debian 11+, Fedora 35+, Arch Linux
- **macOS** : macOS 10.15 (Catalina) ou supérieur
- **Windows** : Windows 10/11 (64-bit)

### Logiciels requis

1. **Python 3.10 ou supérieur**
   - Linux/Mac : Généralement pré-installé ou disponible via gestionnaire de paquets
   - Windows : [Télécharger depuis python.org](https://www.python.org/downloads/)
   - **Important** : Sur Windows, cochez "Add Python to PATH" pendant l'installation

2. **FFmpeg** (sera installé automatiquement par le script)
   - Nécessaire pour le traitement vidéo
   - Le script d'installation le détecte et l'installe si manquant

3. **Pour le support GPU NVIDIA** (optionnel)
   - Pilotes NVIDIA à jour
   - CUDA Toolkit 12.1 ou supérieur
   - [Télécharger CUDA](https://developer.nvidia.com/cuda-downloads)

## Installation automatique

### Linux / macOS

```bash
# Cloner le dépôt
git clone https://github.com/targuy/autocut.git
cd autocut

# Rendre le script exécutable
chmod +x install.sh

# Lancer l'installation
./install.sh
```

Le script vous demandera de choisir le type d'installation :
- **Option 1** : CPU uniquement (pas d'accélération GPU)
- **Option 2** : NVIDIA CUDA (pour les GPU NVIDIA)
- **Option 3** : Toutes les fonctionnalités (détection automatique du GPU)

### Windows

```cmd
# Cloner le dépôt
git clone https://github.com/targuy/autocut.git
cd autocut

# Lancer l'installation
install.bat
```

Le script vous proposera les mêmes options d'installation.

### Ce que fait le script d'installation

1. ✓ Vérifie la version de Python (≥ 3.10)
2. ✓ Détecte et installe FFmpeg si nécessaire
   - Linux : utilise apt-get, yum, dnf, ou pacman
   - macOS : utilise Homebrew
   - Windows : propose imageio-ffmpeg
3. ✓ Crée un environnement virtuel Python isolé
4. ✓ Met à jour pip vers la dernière version
5. ✓ Installe les dépendances appropriées (CPU ou GPU)
6. ✓ Crée un script de lancement `run.sh` ou `run.bat`

## Installation manuelle

Si vous préférez installer manuellement ou si le script automatique échoue :

### Avec Poetry

```bash
# Installer Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Cloner et entrer dans le dépôt
git clone https://github.com/targuy/autocut.git
cd autocut

# Installation CPU
poetry install --extras cpu

# OU installation CUDA
poetry install --extras cuda

# Activer l'environnement
poetry shell
```

### Avec pip et venv

```bash
# Cloner le dépôt
git clone https://github.com/targuy/autocut.git
cd autocut

# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate.bat  # Windows

# Mettre à jour pip
pip install --upgrade pip

# Installation CPU
pip install -e ".[cpu]"

# OU installation CUDA complète
pip install -e ".[cpu]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

## Résolution des conflits GPU/CPU

AutoCutVideo utilise des bibliothèques qui ont des versions séparées pour CPU et GPU. Le projet gère ces conflits via les **extras Poetry**.

### Packages concernés

1. **PyTorch** (`torch` et `torchvision`)
   - Version CPU : Installation par défaut
   - Version CUDA : Nécessite l'index PyTorch CUDA
   - **Conflit** : Ne peuvent pas coexister dans le même environnement

2. **ONNX Runtime**
   - `onnxruntime` : Version CPU uniquement
   - `onnxruntime-gpu` : Version avec support CUDA
   - **Conflit** : Packages mutuellement exclusifs avec le même numéro de version

### Solution implémentée

Le fichier `pyproject.toml` définit des **extras** pour gérer ces conflits :

```toml
[tool.poetry.extras]
cpu = ["torch", "torchvision", "onnxruntime"]
cuda = ["torch", "torchvision", "onnxruntime-gpu"]
ffmpeg = ["imageio-ffmpeg"]
all = ["torch", "torchvision", "onnxruntime", "imageio-ffmpeg"]
```

### Stratégie d'installation

1. **Installation CPU** : Installe directement les versions CPU
   ```bash
   poetry install --extras cpu
   ```

2. **Installation CUDA** : Procédure en deux étapes
   ```bash
   # Étape 1 : Installer les dépendances de base
   poetry install --extras cpu
   
   # Étape 2 : Remplacer par les versions GPU
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip uninstall -y onnxruntime && pip install onnxruntime-gpu
   ```

Cette approche garantit qu'il n'y a jamais de conflit entre versions CPU et GPU.

## Dépannage

### Problème : Python non trouvé

**Symptôme** : `python3: command not found` ou `python: command not found`

**Solution** :
- **Linux** : `sudo apt-get install python3` (Ubuntu/Debian)
- **macOS** : `brew install python3`
- **Windows** : Télécharger depuis [python.org](https://www.python.org/downloads/) et cocher "Add to PATH"

### Problème : FFmpeg non trouvé après installation

**Symptôme** : `ffmpeg: command not found`

**Solution Linux** :
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# Fedora
sudo dnf install ffmpeg

# Arch
sudo pacman -S ffmpeg
```

**Solution macOS** :
```bash
# Installer Homebrew si nécessaire
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer ffmpeg
brew install ffmpeg
```

**Solution Windows** :
1. Télécharger FFmpeg depuis [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extraire l'archive
3. Ajouter le dossier `bin` au PATH système
4. OU utiliser imageio-ffmpeg : `pip install imageio-ffmpeg`

### Problème : Erreur CUDA lors de l'exécution

**Symptôme** : `RuntimeError: CUDA out of memory` ou `CUDA not available`

**Solutions** :
1. Vérifier que les pilotes NVIDIA sont installés : `nvidia-smi`
2. Vérifier que CUDA est disponible :
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```
3. Si CUDA n'est pas disponible, réinstaller avec la version CPU :
   ```bash
   pip uninstall torch torchvision onnxruntime-gpu
   pip install torch torchvision onnxruntime
   ```

### Problème : Conflit de versions de packages

**Symptôme** : `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Solution** :
```bash
# Nettoyer l'environnement
deactivate  # Si dans un venv
rm -rf venv/  # Supprimer l'environnement
python3 -m venv venv  # Recréer
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[cpu]"  # Réinstaller proprement
```

### Problème : Permission denied sur install.sh

**Symptôme** : `bash: ./install.sh: Permission denied`

**Solution** :
```bash
chmod +x install.sh
./install.sh
```

### Problème : ModuleNotFoundError après installation

**Symptôme** : `ModuleNotFoundError: No module named 'xxx'`

**Solution** :
1. Vérifier que l'environnement virtuel est activé
2. Réinstaller les dépendances :
   ```bash
   pip install -e ".[cpu]"
   ```

## Vérification de l'installation

Après installation, exécutez ces commandes pour vérifier que tout fonctionne :

### 1. Vérifier Python
```bash
python3 --version
# Devrait afficher : Python 3.10.x ou supérieur
```

### 2. Vérifier FFmpeg
```bash
ffmpeg -version
# Devrait afficher la version de FFmpeg
```

### 3. Vérifier PyTorch
```python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Affiche la version de PyTorch et la disponibilité CUDA
```

### 4. Vérifier ONNX Runtime
```python
python3 -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}'); print(f'Providers: {onnxruntime.get_available_providers()}')"
# Affiche la version et les providers disponibles (CUDAExecutionProvider pour GPU)
```

### 5. Vérifier les dépendances principales
```bash
pip list | grep -E "torch|ultralytics|opencv|numpy"
```

### 6. Test de l'application
```bash
# Activer l'environnement
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate.bat  # Windows

# Lancer l'application (devrait afficher l'aide)
python main.py --help
```

## Support et aide

Si vous rencontrez des problèmes non couverts par ce guide :

1. Vérifiez les [Issues GitHub](https://github.com/targuy/autocut/issues)
2. Créez une nouvelle issue avec :
   - Votre système d'exploitation
   - Version de Python (`python --version`)
   - Sortie de `pip list`
   - Message d'erreur complet
   - Étapes pour reproduire le problème

## Mise à jour

Pour mettre à jour AutoCutVideo vers la dernière version :

```bash
cd autocut
git pull origin main

# Réactiver l'environnement
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate.bat  # Windows

# Mettre à jour les dépendances
pip install -e ".[cpu]" --upgrade
```

## Désinstallation

Pour désinstaller complètement AutoCutVideo :

```bash
cd autocut
deactivate  # Si dans l'environnement virtuel
rm -rf venv/  # Supprimer l'environnement virtuel
cd ..
rm -rf autocut/  # Supprimer le dépôt
```
