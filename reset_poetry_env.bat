@echo off

REM -- Étape 1 : désactiver conda si actif
CALL conda deactivate

REM -- Étape 2 : aller dans ton dossier de projet
cd /d E:\DocumentsBenoit\pythonProject\AutoCutVideo

REM -- Étape 3 : supprimer l'ancien environnement Poetry s'il existe
poetry env remove python

REM -- Étape 4 : recréer un nouvel environnement avec Python 3.10
poetry env use python3.10

REM -- Étape 5 : installer les dépendances du projet
poetry install

REM -- Étape 6 : installer torch et torchvision GPU
poetry run pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

REM -- Étape 7 : test GPU
poetry run python -c "import torch; print('Torch:', torch.__version__, '| CUDA dispo:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

pause
