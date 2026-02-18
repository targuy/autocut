# Spécifications — Guitar Music Fingering Agent

## 1. Objectif du projet

Développer un outil Python capable de lire des fichiers musicaux pour guitare dans différents formats, et d'y insérer automatiquement les annotations de doigté (numéros de doigts de la main gauche) en utilisant des algorithmes d'optimisation ergonomique.

## 2. Périmètre fonctionnel

### 2.1. Formats d'entrée supportés

| Format | Extensions | Bibliothèque | Priorité | Support doigté natif |
|--------|-----------|---------------|----------|---------------------|
| Guitar Pro 3-5 | `.gp3`, `.gp4`, `.gp5` | PyGuitarPro | **P0** (Phase 1) | ✅ Oui (`leftHandFinger`) |
| MIDI | `.mid`, `.midi` | pretty-midi | **P1** (Phase 1) | ❌ Non |
| MusicXML | `.xml`, `.musicxml`, `.mxl` | music21 | **P1** (Phase 1) | ✅ Oui (`<fingering>`) |
| Guitar Pro 6-8 | `.gpx`, `.gp` | DIY / TuxGuitar export | **P2** (Phase 2) | ✅ Oui (XML interne) |
| Tablature ASCII | `.txt`, `.tab` | Parser custom | **P3** (Phase 3) | ❌ Non |
| PDF (notation standard) | `.pdf` | OMR + music21 | **P4** (Phase 4) | Variable |

### 2.2. Format d'entrée recommandé

**Guitar Pro (.gp5)** est le format prioritaire car il :
- Contient les informations de corde et de case pour chaque note
- Supporte nativement les annotations de doigté (`leftHandFinger`, `rightHandFinger`)
- Est bien supporté par la bibliothèque open-source PyGuitarPro (lecture et écriture)
- Est le format le plus utilisé dans les communautés de guitaristes

**MusicXML** est le second format recommandé car il :
- Est un standard ouvert (W3C) pour la notation musicale
- Supporte les éléments `<technical><fingering>` pour les doigtés
- Est importable/exportable par la plupart des logiciels de notation (MuseScore, Finale, Sibelius)
- Est bien supporté par music21

### 2.3. Formats de sortie

| Format | Annotation doigté | Méthode |
|--------|------------------|---------|
| Guitar Pro (.gp5) | `leftHandFinger` sur chaque note | PyGuitarPro |
| MusicXML (.xml) | `<technical><fingering>` | music21 |
| JSON (rapport) | Données structurées | Sérialisation Python |

### 2.4. Fonctionnalités principales

#### Phase 1 — MVP (Minimum Viable Product)

- [ ] **F1** : Lire un fichier Guitar Pro (.gp3/.gp4/.gp5) et extraire les notes avec corde/case
- [ ] **F2** : Lire un fichier MIDI et convertir les notes en positions guitare
- [ ] **F3** : Lire un fichier MusicXML et extraire les notes/doigtés existants
- [ ] **F4** : Calculer le doigté optimal par programmation dynamique (DP)
- [ ] **F5** : Exporter le résultat vers Guitar Pro (.gp5) avec doigté intégré
- [ ] **F6** : Exporter le résultat vers MusicXML avec éléments `<fingering>`
- [ ] **F7** : Interface en ligne de commande (CLI)

#### Phase 2 — Améliorations

- [ ] **F8** : Support des accords (doigté simultané de plusieurs notes)
- [ ] **F9** : Support des accordages alternatifs (Drop D, DADGAD, etc.)
- [ ] **F10** : Support du capodastre
- [ ] **F11** : Paramètres ergonomiques personnalisables (taille de la main, etc.)
- [ ] **F12** : Support des fichiers Guitar Pro 6-8 (.gpx/.gp)

#### Phase 3 — Avancé

- [ ] **F13** : Support des tablatures ASCII
- [ ] **F14** : Import depuis des bases de données en ligne (Ultimate Guitar, etc.)
- [ ] **F15** : Utilisation de modèles ML/DL pour l'optimisation du doigté
- [ ] **F16** : Interface web ou GUI
- [ ] **F17** : Support de la main droite (doigté classique p-i-m-a)

## 3. Spécifications techniques

### 3.1. Modèle de données

```
Note
├── midi_pitch: int (0-127)
├── start_time: float
├── duration: float
└── velocity: int

GuitarNote (hérite de Note conceptuellement)
├── midi_pitch: int
├── string: int (1-6)
├── fret: int (0-24)
├── finger: int | None (0=ouvert, 1=index, 2=majeur, 3=annulaire, 4=auriculaire)
├── start_time: float
└── duration: float

Chord
├── notes: list[GuitarNote]
├── start_time: float
└── duration: float

Measure
├── number: int
├── chords: list[Chord]
└── tempo: float | None

Track
├── name: str
├── measures: list[Measure]
├── tuning: list[int]  # Notes MIDI des cordes à vide
├── capo: int
└── tempo: float
```

### 3.2. Pipeline de traitement

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Fichier    │────▶│   Parser     │────▶│  Fingering   │────▶│  Exporter    │
│   d'entrée   │     │  (GP/MIDI/   │     │  Optimizer   │     │  (GP/XML)    │
│  (.gp5/.mid/ │     │   XML)       │     │  (DP/Viterbi)│     │              │
│   .xml)      │     └──────────────┘     └──────────────┘     └──────────────┘
└──────────────┘            │                    │                     │
                            ▼                    ▼                     ▼
                      Track (sans          Track (avec           Fichier
                       doigté)              doigté)              annoté
```

### 3.3. Algorithme de doigté (DP)

L'algorithme utilise la programmation dynamique de type Viterbi :

1. **États** : Chaque état est un tuple `(corde, case, doigt)` pour une note donnée
2. **Transitions** : Coût de passage d'un état à un autre (ergonomie)
3. **Fonction de coût** : Somme pondérée de critères :
   - Distance de déplacement sur le manche (cases) : poids 1.0
   - Changement de corde : poids 0.5
   - Écart entre doigts > seuil : pénalité 3.0
   - Position haute sur le manche (> case 12) : pénalité 0.2/case
   - Corde à vide : bonus -0.5
4. **Objectif** : Minimiser le coût total du chemin

### 3.4. Conventions de doigté

| Doigt | Valeur | Main gauche | Main droite (classique) |
|-------|--------|-------------|------------------------|
| Ouvert | 0 | Corde à vide | — |
| Index | 1 | 1er doigt | i (indice) |
| Majeur | 2 | 2ème doigt | m (medio) |
| Annulaire | 3 | 3ème doigt | a (anular) |
| Auriculaire | 4 | 4ème doigt | — |
| Pouce | — | Rare (barre) | p (pulgar) |

### 3.5. Accordage standard

```
Corde 1 (aigu) : E4 = MIDI 64
Corde 2         : B3 = MIDI 59
Corde 3         : G3 = MIDI 55
Corde 4         : D3 = MIDI 50
Corde 5         : A2 = MIDI 45
Corde 6 (grave) : E2 = MIDI 40
```

## 4. Exigences non fonctionnelles

### 4.1. Performance
- Le calcul du doigté pour un morceau de 200 mesures doit prendre < 5 secondes
- L'algorithme DP est en O(N × K²) où N = nombre de notes et K = candidats par note

### 4.2. Qualité
- Tests unitaires pour chaque module (pytest)
- Couverture de code > 80%
- Linting avec flake8, formatage avec black

### 4.3. Compatibilité
- Python 3.10+
- Gestion des dépendances avec Poetry
- Compatible Linux, macOS, Windows

### 4.4. Extensibilité
- Architecture modulaire (parsers, optimiseurs, exporteurs interchangeables)
- Possibilité d'ajouter de nouveaux formats facilement
- Possibilité de remplacer l'algorithme DP par un modèle ML/DL

## 5. Dépendances

### 5.1. Dépendances de production

| Package | Version | Usage |
|---------|---------|-------|
| `pyguitarpro` | ^0.10 | Lecture/écriture Guitar Pro |
| `music21` | ^9.1 | Manipulation MusicXML |
| `pretty-midi` | ^0.2.10 | Lecture MIDI |
| `numpy` | ^1.26 | Calculs numériques |

### 5.2. Dépendances de développement

| Package | Version | Usage |
|---------|---------|-------|
| `pytest` | ^8.3 | Tests unitaires |
| `flake8` | ^7.2 | Linting |
| `black` | ^25.1 | Formatage |

## 6. Risques et limitations

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| Ambiguïté du doigté (plusieurs solutions valides) | Moyen | Le DP choisit la solution de coût minimal ; possibilité de proposer des alternatives |
| Accords complexes (barré, etc.) | Élevé | Phase 2 : extension du modèle pour les accords |
| Formats Guitar Pro récents (.gpx/.gp) non supportés par PyGuitarPro | Moyen | Conversion préalable via TuxGuitar ou parser XML custom |
| Performance sur de très longs morceaux | Faible | Optimisation du DP avec pruning des candidats |
| Techniques spéciales (slide, hammer-on, pull-off) | Moyen | Phase 2 : intégration des effets dans le coût |
