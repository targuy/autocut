# 🎸 Guitar Music Fingering Agent

**Annotation automatique du doigté pour guitare à partir de fichiers musicaux**

## Description

Guitar Music Fingering Agent est un outil Python qui génère automatiquement les annotations de doigté (numéros de doigts de la main gauche : 0=ouvert, 1=index, 2=majeur, 3=annulaire, 4=auriculaire) pour des morceaux de guitare à partir de différentes sources de données musicales.

Le projet prend en charge plusieurs formats d'entrée :
- **Guitar Pro** (.gp3, .gp4, .gp5) — format principal recommandé
- **MIDI** (.mid, .midi)
- **MusicXML** (.xml, .musicxml)
- **Tablatures ASCII** (.txt) — support futur

## Pourquoi ce projet ?

La plupart des partitions et tablatures de guitare disponibles (Guitar Pro, MIDI, tablatures en ligne) indiquent les notes à jouer (corde + case) mais **ne précisent pas quel doigt utiliser**. Cette information est pourtant essentielle pour :
- Jouer de manière ergonomique et fluide
- Éviter les tensions et blessures
- Faciliter l'apprentissage des morceaux
- Permettre les transitions optimales entre positions

## Architecture

```
guitar_music_fingering_agent/
├── README.md                    # Ce fichier
├── SPECIFICATIONS.md            # Spécifications fonctionnelles et techniques
├── STATE_OF_THE_ART.md          # État de l'art des solutions existantes
├── pyproject.toml               # Configuration du projet (Poetry)
├── src/
│   └── guitar_fingering/
│       ├── __init__.py          # Point d'entrée du package
│       ├── parsers/             # Lecture des fichiers musicaux
│       │   ├── __init__.py
│       │   ├── base.py          # Classe de base abstraite
│       │   ├── guitarpro.py     # Parser Guitar Pro (.gp3/.gp4/.gp5)
│       │   ├── midi.py          # Parser MIDI
│       │   └── musicxml.py      # Parser MusicXML
│       ├── fingering/           # Algorithmes de calcul du doigté
│       │   ├── __init__.py
│       │   ├── engine.py        # Moteur principal de doigté
│       │   ├── cost.py          # Fonctions de coût ergonomique
│       │   └── optimizer.py     # Optimisation par programmation dynamique
│       ├── exporters/           # Export des résultats
│       │   ├── __init__.py
│       │   ├── base.py          # Classe de base abstraite
│       │   ├── guitarpro.py     # Export vers Guitar Pro
│       │   └── musicxml.py      # Export vers MusicXML
│       ├── models/              # Modèles de données
│       │   ├── __init__.py
│       │   └── music.py         # Note, Chord, Measure, etc.
│       └── utils/               # Utilitaires
│           ├── __init__.py
│           └── guitar.py        # Constantes et helpers guitare
├── tests/                       # Tests unitaires
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_guitar_utils.py
│   └── test_fingering.py
├── docs/                        # Documentation additionnelle
└── data/
    └── samples/                 # Fichiers d'exemple pour les tests
```

## Installation

```bash
cd guitar_music_fingering_agent
pip install -e .
```

### Dépendances principales

- `pyguitarpro` — Lecture/écriture des fichiers Guitar Pro (.gp3/.gp4/.gp5)
- `music21` — Manipulation de données musicales et export MusicXML
- `pretty-midi` — Lecture des fichiers MIDI
- `numpy` — Calculs numériques pour l'optimisation

## Utilisation rapide

```python
from guitar_fingering import FingeringEngine

engine = FingeringEngine()

# Charger un fichier Guitar Pro
engine.load('mon_morceau.gp5')

# Calculer le doigté optimal
engine.compute_fingering()

# Exporter avec le doigté annoté
engine.export('mon_morceau_avec_doigte.gp5')
```

## Formats supportés

| Format       | Lecture | Écriture doigté | Priorité |
|-------------|---------|-----------------|----------|
| Guitar Pro 3-5 | ✅      | ✅               | Haute    |
| MIDI         | ✅      | ❌ (export MusicXML) | Moyenne  |
| MusicXML     | ✅      | ✅               | Moyenne  |
| ASCII Tab    | 🔜      | 🔜               | Future   |

## Algorithme de doigté

Le moteur de doigté utilise une approche par **programmation dynamique** (DP) inspirée des travaux de recherche en optimisation du doigté pour instruments à cordes :

1. **Extraction des notes** : Lecture du fichier source et extraction de la séquence de notes/accords
2. **Génération des candidats** : Pour chaque note, génération de toutes les combinaisons (corde, case, doigt) possibles
3. **Calcul du coût** : Évaluation ergonomique de chaque transition entre positions
4. **Optimisation DP** : Recherche du chemin optimal minimisant le coût total

Voir [STATE_OF_THE_ART.md](STATE_OF_THE_ART.md) pour les détails des algorithmes et des références académiques.

## Licence

MIT

## Auteur

Benoit Guitard
