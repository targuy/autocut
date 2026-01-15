# Analyse des Conflits de Dépendances GPU/CPU

## Résumé

Ce document analyse les conflits potentiels entre les packages Python qui ont des versions séparées pour CPU et GPU, et explique la solution implémentée dans AutoCutVideo.

## Packages concernés

### 1. PyTorch (torch et torchvision)

#### Le problème

PyTorch distribue des wheels différents selon le backend :
- **CPU** : `torch==2.x.x` depuis PyPI standard
- **CUDA 11.8** : `torch==2.x.x+cu118` depuis l'index PyTorch
- **CUDA 12.1** : `torch==2.x.x+cu121` depuis l'index PyTorch
- **ROCm** (AMD) : `torch==2.x.x+rocm5.x` depuis l'index PyTorch

**Problème** : Même numéro de version base, mais wheels incompatibles. Si installé depuis le mauvais index, PyTorch peut :
- Ne pas détecter le GPU
- Avoir des performances réduites
- Échouer au runtime avec des erreurs CUDA

#### Impact

```python
# Version CPU installée par erreur
>>> import torch
>>> torch.cuda.is_available()
False  # Même avec un GPU NVIDIA présent

# Version CUDA installée correctement
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'NVIDIA GeForce RTX 3080'
```

### 2. ONNX Runtime

#### Le problème

ONNX Runtime a deux packages séparés et mutuellement exclusifs :
- **onnxruntime** : Version CPU uniquement
- **onnxruntime-gpu** : Version avec support CUDA et TensorRT

**Problème critique** : Les deux packages ont le même numéro de version (ex: `1.17.0`) mais sont mutuellement exclusifs. Installer les deux crée des conflits :

```bash
# Conflit si les deux sont installés
pip install onnxruntime==1.17.0
pip install onnxruntime-gpu==1.17.0  # CONFLIT!
# ERROR: Cannot uninstall 'onnxruntime'
```

#### Impact

```python
# Version CPU
>>> import onnxruntime
>>> onnxruntime.get_available_providers()
['CPUExecutionProvider']

# Version GPU
>>> import onnxruntime
>>> onnxruntime.get_available_providers()
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 3. Autres packages à surveiller

#### OpenCV

OpenCV a plusieurs variantes :
- `opencv-python` : Version standard
- `opencv-contrib-python` : Avec modules contrib
- `opencv-python-headless` : Sans GUI (serveurs)

**Pas de conflit GPU/CPU** mais peuvent se chevaucher.

#### TensorFlow (non utilisé actuellement)

Si ajouté à l'avenir :
- `tensorflow` : CPU uniquement depuis ~2.11
- `tensorflow[and-cuda]` : Avec support GPU (nouvelle méthode)
- ~~`tensorflow-gpu`~~ : Déprécié

## Solution implémentée

### Architecture des extras Poetry

```toml
[tool.poetry.dependencies]
# Packages optionnels
torch = {version = "*", optional = true}
torchvision = {version = "*", optional = true}
onnxruntime = {version = "^1.17.0", optional = true}
onnxruntime-gpu = {version = "^1.17.0", optional = true}
imageio-ffmpeg = {version = "^0.5.1", optional = true}

[tool.poetry.extras]
cpu = ["torch", "torchvision", "onnxruntime"]
cuda = ["torch", "torchvision", "onnxruntime-gpu"]
ffmpeg = ["imageio-ffmpeg"]
all = ["torch", "torchvision", "onnxruntime", "imageio-ffmpeg"]
```

### Stratégie d'installation

#### Pour CPU uniquement
```bash
poetry install --extras cpu
```

Installe :
- `torch` (CPU depuis PyPI)
- `torchvision` (CPU)
- `onnxruntime` (CPU)

#### Pour CUDA (installation en deux étapes)

**Étape 1** : Installer les dépendances de base
```bash
poetry install --extras cpu
```

**Étape 2** : Remplacer par les versions GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

**Pourquoi deux étapes ?**
1. Poetry ne supporte pas les index URL personnalisés dans pyproject.toml
2. Permet de résoudre d'abord toutes les dépendances compatibles
3. Évite les conflits entre onnxruntime et onnxruntime-gpu

### Scripts d'installation automatiques

Les scripts `install.sh` et `install.bat` implémentent cette logique :

```bash
# Extrait du script d'installation
case $INSTALL_CHOICE in
    1)  # CPU
        pip install -e ".[cpu]"
        ;;
    2)  # CUDA
        pip install -e ".[cpu]"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip uninstall -y onnxruntime
        pip install onnxruntime-gpu
        ;;
    3)  # Auto-détection
        pip install -e ".[all]"
        if command -v nvidia-smi &> /dev/null; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            pip uninstall -y onnxruntime
            pip install onnxruntime-gpu
        fi
        ;;
esac
```

## Tests de validation

### Test 1 : Pas de conflit dans l'environnement

```bash
pip list | grep -E "torch|onnxruntime"
# CPU : devrait montrer torch et onnxruntime
# CUDA : devrait montrer torch et onnxruntime-gpu, PAS onnxruntime
```

### Test 2 : Détection GPU correcte

```python
import torch
import onnxruntime

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ONNX providers: {onnxruntime.get_available_providers()}")

# CPU attendu :
# CUDA available: False
# ONNX providers: ['CPUExecutionProvider']

# CUDA attendu :
# CUDA available: True
# ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Test 3 : Pas de dépendances en double

```bash
pip check
# Devrait afficher : "No broken requirements found."
```

## Comparaison avec d'autres approches

### Approche 1 : Fichiers requirements séparés (non utilisée)

```
requirements-cpu.txt
requirements-cuda.txt
```

**Avantages** :
- Simple
- Pas de dépendance à Poetry

**Inconvénients** :
- Duplication du code
- Difficile à maintenir
- Pas de résolution de dépendances automatique

### Approche 2 : Détection automatique au runtime (non utilisée)

```python
import torch
if torch.cuda.is_available():
    import onnxruntime_gpu as ort
else:
    import onnxruntime as ort
```

**Avantages** :
- Flexibilité maximale

**Inconvénients** :
- Les deux packages doivent être installés → conflit !
- Code plus complexe
- Erreurs difficiles à déboguer

### Approche 3 : Extras Poetry (utilisée ✓)

**Avantages** :
- Déclaratif et maintenable
- Résolution de dépendances automatique
- Un seul fichier de configuration
- Compatible avec pip et Poetry

**Inconvénients** :
- Nécessite une installation en deux étapes pour CUDA
- Utilisateurs doivent choisir explicitement

## Recommandations pour l'avenir

### Ajout de nouveaux packages

Lors de l'ajout de packages avec variants GPU/CPU :

1. **Déclarer comme optionnel** dans `[tool.poetry.dependencies]`
2. **Ajouter aux extras appropriés** dans `[tool.poetry.extras]`
3. **Mettre à jour les scripts d'installation** pour gérer l'installation en deux étapes si nécessaire
4. **Documenter** dans ce fichier et INSTALL.md
5. **Tester** les deux variants (CPU et GPU)

### Tests automatisés

Implémenter des tests CI/CD pour :
- ✓ Installation CPU réussit
- ✓ Installation CUDA réussit (si runner GPU disponible)
- ✓ Pas de conflits de packages (`pip check`)
- ✓ Import des modules critiques réussit

### Monitoring des versions

Surveiller les mises à jour de :
- PyTorch (changements d'index CUDA)
- ONNX Runtime (nouvelles versions GPU)
- Ultralytics (dépendances PyTorch)

## Références

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [Poetry Dependency Groups](https://python-poetry.org/docs/managing-dependencies/)
- [pip Index URLs](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-index-url)

## Historique

- **2025-01-15** : Implémentation initiale avec extras Poetry
- Identification des conflits torch/onnxruntime
- Création des scripts d'installation automatiques
- Documentation complète
