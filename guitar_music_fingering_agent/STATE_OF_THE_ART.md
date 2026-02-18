# État de l'art — Doigté automatique pour guitare

## Table des matières

1. [Introduction](#1-introduction)
2. [Approches algorithmiques classiques](#2-approches-algorithmiques-classiques)
3. [Approches par apprentissage automatique](#3-approches-par-apprentissage-automatique)
4. [Approches par LLM et IA générative](#4-approches-par-llm-et-ia-générative)
5. [Bibliothèques et outils logiciels](#5-bibliothèques-et-outils-logiciels)
6. [Formats de fichiers musicaux](#6-formats-de-fichiers-musicaux)
7. [Jeux de données](#7-jeux-de-données)
8. [Synthèse et choix pour le projet](#8-synthèse-et-choix-pour-le-projet)
9. [Références bibliographiques](#9-références-bibliographiques)

---

## 1. Introduction

Le problème du doigté automatique pour guitare consiste à déterminer, pour une séquence de notes donnée, le doigt optimal à utiliser sur chaque note afin de minimiser la difficulté de jeu. Ce problème est fondamentalement **combinatoire** : pour chaque note MIDI, il existe potentiellement plusieurs positions (corde, case) sur le manche, et pour chaque position, plusieurs doigts possibles.

La recherche dans ce domaine remonte aux années 1980 et a évolué à travers plusieurs paradigmes : de l'optimisation classique (programmation dynamique, graphes) vers les modèles probabilistes (HMM, Viterbi), puis vers l'apprentissage profond et les grands modèles de langage (LLM).

---

## 2. Approches algorithmiques classiques

### 2.1. Programmation dynamique (DP)

La programmation dynamique est l'approche la plus étudiée et la plus efficace pour le doigté monophonique (notes jouées une par une).

#### Sayegh (1989) — Travail fondateur

- **Référence** : S. I. Sayegh, *"Fingering for String Instruments with the Optimum Path Paradigm"*, Computer Music Journal, Vol. 13, No. 3, 1989.
- **Approche** : Formulation du problème comme la recherche du chemin de coût minimal dans un graphe acyclique dirigé (DAG), où chaque nœud représente une position (corde, case, doigt) et chaque arête un coût de transition.
- **Fonction de coût** : Basée sur la distance physique entre positions, les contraintes biomécaniques de la main, et les préférences stylistiques.
- **Complexité** : O(N × K²) où N = nombre de notes et K = candidats par note.
- **Limitation** : Conçu pour les instruments à cordes en général, pas spécifiquement optimisé pour la guitare.

#### Radicioni et Lombardo (2004) — Recherche en graphe pour guitare

- **Approche** : Extension du paradigme de Sayegh avec des contraintes spécifiques à la guitare (6 cordes, position du pouce, barré).
- **Contribution** : Modélisation plus fine des contraintes ergonomiques et des transitions entre positions.

#### Radisavljevic et Driessen (2004) — Apprentissage de la fonction de coût

- **Référence** : *"Path Difference Learning for Guitar Fingering Problem"*, ICMC 2004.
- **Approche** : Utilisation du DP pour la structure, mais **apprentissage automatique de la fonction de coût** à partir de données réelles de performance (doigtés joués par des guitaristes experts).
- **Méthode** : Descente de gradient pour ajuster les poids de la fonction de coût.
- **Innovation** : Première utilisation de données réelles pour calibrer l'optimisation.

#### Grozman et Norman (2013) — Algorithme optimal pour guitare

- **Référence** : *"An Algorithm for Optimal Guitar Fingering"*, KTH Royal Institute of Technology, 2013.
- **Approche** : DP avec fonction de coût biomécanique tenant compte de l'aisance du mouvement des doigts, des contraintes physiques de la main, et de la réduction de l'espace de recherche.
- **Validation** : Tests avec des professeurs de guitare confirmant la jouabilité des doigtés générés.
- **Limitation** : Évaluation locale (paires de notes adjacentes), manque de vision globale du contexte.

#### Itoh et Hayashida (2004) — Distance de Manhattan

- **Référence** : *"Optimization for Guitar Fingering on Single Notes"*, 2004.
- **Approche** : Formulation comme problème de décision multi-étapes, utilisant la distance de Manhattan (déplacement en cases + changement de corde) comme fonction de coût.
- **Application** : Conçu aussi pour les robots guitaristes et les outils d'enseignement assisté par ordinateur.

### 2.2. Modèles de Markov cachés (HMM) et algorithme de Viterbi

#### Hori et Sagayama (2016) — Minimax Viterbi

- **Référence** : *"Minimax Viterbi Algorithm for HMM-based Guitar Fingering Decision"*, ISMIR 2016.
- **Approche** : Modélisation du problème comme un HMM où les **notes sont les observations** et les **positions de la main sont les états cachés**.
- **Innovation** : Variante **minimax** de l'algorithme de Viterbi qui minimise la difficulté **maximale** (plutôt que la somme des difficultés), ce qui est plus adapté aux débutants.
- **Avantage** : Évite les passages localement très difficiles même si le coût total est légèrement plus élevé.

### 2.3. Recherche par contraintes et satisfaction

- Certains travaux modélisent le doigté comme un **problème de satisfaction de contraintes** (CSP) :
  - Contraintes dures : la main ne peut pas s'étendre au-delà d'une certaine amplitude, deux doigts ne peuvent pas être sur la même case (sauf barré), etc.
  - Contraintes souples : préférence pour les positions basses, éviter les sauts de position, etc.
- Les solveurs CSP (comme Google OR-Tools) peuvent être utilisés pour explorer l'espace des solutions.

### 2.4. Synthèse des approches algorithmiques

| Méthode | Complexité | Avantages | Limitations |
|---------|-----------|-----------|-------------|
| DP / Plus court chemin | O(N × K²) | Rapide, optimal, bien compris | Fonction de coût à calibrer manuellement |
| HMM / Viterbi | O(N × K²) | Cadre probabiliste, extensible | Nécessite des données d'entraînement |
| Minimax Viterbi | O(N × K²) | Évite les pics de difficulté | Coût total parfois sous-optimal |
| CSP | Variable | Expressif, contraintes explicites | Peut être lent pour de grandes instances |

---

## 3. Approches par apprentissage automatique

### 3.1. Réseaux de neurones pour la transcription guitare

#### TabCNN (Wiggins, 2019)

- **Approche** : Réseau de neurones convolutif (CNN) pour la transcription audio → tablature.
- **Entrée** : Spectrogramme audio (CQT — Constant-Q Transform).
- **Sortie** : Pour chaque pas de temps, la case jouée sur chaque corde (0-19 ou silence).
- **Limitation** : Prédit les positions mais pas les doigts.

#### TabInception (2023)

- **Approche** : Extension de TabCNN avec des modules Inception pour une meilleure extraction de caractéristiques multi-échelles.
- **Amélioration** : Surpasse TabCNN sur la précision de la transcription tablature.

#### TPMNet (2025)

- **Référence** : *"Multi-task learning-based temporal pattern matching network for guitar tablature transcription"*, Springer, 2025.
- **Approche** : Réseau multi-tâches combinant détection de pitch et assignation corde/case.
- **Innovation** : Utilisation de correspondance de patterns temporels pour une meilleure cohérence.

### 3.2. Modèles de séquence (RNN, LSTM, Transformer)

#### TART — Technique-Aware Audio-to-Tab (Gupta et al., 2025)

- **Référence** : *"TART: A Comprehensive Tool for Technique-Aware Audio-to-Tab Guitar Transcription"*, UC Berkeley, 2025.
- **Pipeline en 4 étapes** :
  1. Détection multi-pitch (audio → notes)
  2. Classification des techniques (bend, slide, hammer-on, pull-off)
  3. Assignation corde/case via Transformer
  4. Génération de tablature via LSTM
- **Innovation** : Premier système intégrant les techniques de jeu dans la transcription.

#### Attention-based Guitar Transcription (Kim, Hayashi et al., 2022)

- **Référence** : *"Note-level Automatic Guitar Transcription Using Attention Mechanism"*, EURASIP 2022.
- **Approche** : Mécanisme d'attention pour capturer les dépendances temporelles et contextuelles.
- **Avantage** : Meilleure généralisation que les CNN seuls.

### 3.3. Modèles combinant DP et ML

#### Bontempi et al. (2024) — MIDI vers tablature riche

- **Référence** : *"From MIDI to Rich Tablatures: an Automatic Generative System incorporating Lead Guitarists' Fingering and Stylistic Choices"*, SMC 2024.
- **Approche** : Combine **programmation dynamique** avec **modélisation statistique** des choix stylistiques de guitaristes professionnels.
- **Contribution** : Génère non seulement les positions mais aussi les **doigtés, articulations et techniques expressives**.
- **Export** : MusicXML pour intégration dans les logiciels de notation.
- **Innovation clé** : Utilisation du minimax Viterbi avec des fonctions de coût apprises à partir de données réelles de performance.

#### Kuo et al. (2025) — Estimation de difficulté

- **Référence** : *"A Novel Strategy for Difficulty Estimation in the Guitar Fingering Problem"*, Springer, 2025.
- **Contribution** : Nouvelles fonctions de fitness et stratégies d'estimation de la difficulté pour le doigté de solos.

---

## 4. Approches par LLM et IA générative

### 4.1. Génération de tablature par Transformer

#### Sarmento (2024) — Tablature par Deep Learning

- **Référence** : P. Sarmento, *"Guitar Tablature Generation with Deep Learning"*, Thèse PhD, 2024.
- **Modèles** : Transformer-XL entraîné sur de grands corpus de fichiers Guitar Pro (DadaGP).
- **Variantes** :
  - **ShredGP** : Génération de solos de guitare metal/shred
  - **LooperGP** : Génération de boucles de guitare
  - **ProgGP** : Génération de riffs progressifs
- **Innovation** : Utilisation de tokens de contrôle pour le genre, le style et les techniques.

### 4.2. LLM généralistes pour la musique

#### ChatMusician (2024)

- **Référence** : *"ChatMusician: Understanding and Generating Music Intrinsically with LLM"*, ACL Findings 2024.
- **Approche** : Fine-tuning de LLaMA 2 sur des données musicales en notation ABC.
- **Capacités** : Génération, compréhension et raisonnement musical.
- **Performance** : Surpasse GPT-3.5 sur les tâches de théorie musicale.

#### Midi-LLM (2025)

- **Référence** : *"Midi-LLM: Adapting Large Language Models for Text-to-MIDI Music Generation"*, MIT, 2025.
- **Approche** : Adaptation de LLM généralistes pour générer des fichiers MIDI complets à partir de descriptions en langage naturel.
- **Potentiel** : Convertible en tablature guitare via un pipeline de post-traitement.

### 4.3. Outils commerciaux et communautaires

- **TabMaker + ChatGPT** : Utilisation de GPT-4 pour transcrire des mélodies en tablatures.
- **MuseScore + IA** : Intégration d'outils d'IA pour l'assistance à la notation.
- **Guitar Pro 8** : Fonctionnalités d'auto-tablature (propriétaire).

### 4.4. Limites actuelles des LLM pour le doigté

| Aspect | Évaluation |
|--------|-----------|
| Génération de notes/tabs | ✅ Bon (surtout pour riffs courts) |
| Doigté précis et jouable | ⚠️ Moyen (manque de contraintes biomécaniques) |
| Cohérence sur un morceau complet | ⚠️ Moyen (hallucinations possibles) |
| Techniques expressives | ⚠️ Variable selon le modèle |
| Coût computationnel | ❌ Élevé (inférence GPU) |

---

## 5. Bibliothèques et outils logiciels

### 5.1. Bibliothèques Python pour les fichiers musicaux

| Bibliothèque | Formats | Fonctionnalités | Doigté |
|--------------|---------|-----------------|--------|
| **PyGuitarPro** | .gp3, .gp4, .gp5 | Lecture/écriture complète | ✅ `leftHandFinger`, `rightHandFinger` |
| **music21** | MusicXML, MIDI, ABC, etc. | Analyse musicale complète | ✅ `Fingering`, `StringNumber`, `Fret` |
| **pretty-midi** | .mid, .midi | Lecture/écriture MIDI | ❌ (pas de concept de doigté) |
| **mido** | .mid, .midi | Lecture/écriture MIDI bas niveau | ❌ |
| **mingus** | Théorie musicale | Accords, gammes, intervalles | ❌ |

### 5.2. Bibliothèques pour l'optimisation

| Bibliothèque | Usage | Adapté au doigté |
|--------------|-------|------------------|
| **NumPy** | Calculs matriciels, DP | ✅ Idéal pour les tables DP |
| **SciPy** | Optimisation scientifique | ✅ Fonctions d'optimisation |
| **Google OR-Tools** | CSP, programmation par contraintes | ✅ Pour les approches CSP |
| **NetworkX** | Graphes, plus court chemin | ✅ Pour la modélisation en graphe |

### 5.3. Projets open-source existants

| Projet | Langage | Approche | Lien |
|--------|---------|----------|------|
| **guitar_dp** (jgollub1) | Python | DP pour MIDI monophonique | [GitHub](https://github.com/jgollub1/guitar_dp) |
| **awesome-agt** (lucasgris) | — | Collection de ressources AGT | [GitHub](https://github.com/lucasgris/awesome-agt) |
| **TabCNN** | Python/TF | CNN audio → tablature | Recherche académique |
| **DadaGP** | Dataset | Corpus Guitar Pro pour ML | Recherche académique |

### 5.4. Logiciels de notation avec fonctions de doigté

| Logiciel | Type | Auto-doigté | Export doigté |
|----------|------|-------------|---------------|
| **Guitar Pro 8** | Commercial | Partiel (auto-tab) | ✅ GP5, MusicXML |
| **MuseScore 4** | Open source | ❌ | ✅ MusicXML |
| **TuxGuitar** | Open source | ❌ | ✅ GP5, MusicXML |
| **Finale** | Commercial | ❌ | ✅ MusicXML |
| **Sibelius** | Commercial | ❌ | ✅ MusicXML |
| **LilyPond** | Open source | ❌ | ✅ (notation textuelle) |

---

## 6. Formats de fichiers musicaux

### 6.1. Comparaison des formats pour le doigté

| Critère | Guitar Pro (.gp5) | MusicXML | MIDI | ASCII Tab |
|---------|-------------------|----------|------|-----------|
| Info corde/case | ✅ Natif | ✅ `<string>/<fret>` | ❌ | ✅ Implicite |
| Info doigté | ✅ `leftHandFinger` | ✅ `<fingering>` | ❌ | ❌ |
| Effets/techniques | ✅ Riche | ✅ Partiel | ❌ | Textuel |
| Bibliothèque Python | ✅ PyGuitarPro | ✅ music21 | ✅ pretty-midi | Custom |
| Lecture + écriture | ✅ | ✅ | ✅ | Partiel |
| Communauté | Large (guitaristes) | Standard W3C | Universel | Très large |

### 6.2. Détail du support doigté par format

#### Guitar Pro (.gp5)

Le format Guitar Pro est le plus riche pour la guitare. Chaque note porte :
- `string` : numéro de corde (1-6)
- `value` : numéro de case (0-24)
- `effect.leftHandFinger` : doigt main gauche (enum : open, index, middle, ring, pinky)
- `effect.rightHandFinger` : doigt main droite (enum : open, thumb, index, middle, ring)

Accès via PyGuitarPro :
```python
import guitarpro
song = guitarpro.parse('file.gp5')
note = song.tracks[0].measures[0].voices[0].beats[0].notes[0]
note.effect.leftHandFinger  # guitarpro.models.Fingering
```

#### MusicXML

Le standard MusicXML supporte les doigtés via :
```xml
<note>
  <pitch><step>E</step><octave>4</octave></pitch>
  <notations>
    <technical>
      <string>1</string>
      <fret>0</fret>
      <fingering>0</fingering>
    </technical>
  </notations>
</note>
```

Accès via music21 :
```python
from music21 import note, articulations
n = note.Note('E4')
n.articulations.append(articulations.Fingering(1))
n.articulations.append(articulations.StringNumber(1))
```

#### MIDI

Le format MIDI ne contient **aucune information de position sur le manche ni de doigté**. Il faut :
1. Convertir chaque note MIDI en positions (corde, case) possibles
2. Choisir la meilleure position selon un critère d'optimisation
3. Assigner le doigté via l'algorithme DP

---

## 7. Jeux de données

### 7.1. Datasets disponibles

| Dataset | Contenu | Taille | Usage |
|---------|---------|--------|-------|
| **GuitarSet** | Audio + annotations tablature | 360 extraits | Entraînement/évaluation |
| **DadaGP** | Fichiers Guitar Pro | ~26K morceaux | Entraînement de modèles génératifs |
| **Guitar-TECHS** | Audio + techniques | Variable | Classification des techniques |
| **IDMT-SMT-Guitar** | Audio guitare | 4700+ notes | Détection de pitch |

### 7.2. Sources de tablatures en ligne

| Source | Format | Accès | Volume |
|--------|--------|-------|--------|
| Ultimate Guitar | ASCII, Guitar Pro | API (payante) | 1M+ tabs |
| Songsterr | Guitar Pro | Web | 800K+ |
| 911Tabs | Liens vers tabs | Agrégateur | Variable |
| MuseScore Community | MusicXML | API | 1M+ partitions |

---

## 8. Synthèse et choix pour le projet

### 8.1. Approche retenue pour la Phase 1

Pour le MVP, nous adoptons une approche **DP classique** car :

1. **Bien comprise et éprouvée** : 35+ ans de recherche valident l'approche
2. **Rapide** : O(N × K²) est largement suffisant pour des morceaux de guitare typiques
3. **Pas de données d'entraînement nécessaires** : Fonctionne avec une fonction de coût heuristique
4. **Déterministe** : Résultats reproductibles et explicables
5. **Extensible** : La fonction de coût peut être affinée avec des données réelles (Phase 2+)

### 8.2. Format prioritaire

**Guitar Pro (.gp5)** via PyGuitarPro :
- Seul format offrant à la fois les positions (corde/case) ET le support natif du doigté
- Grande communauté de guitaristes utilisant ce format
- Bibliothèque Python mature pour la lecture ET l'écriture

**MusicXML** en second :
- Standard ouvert et interopérable
- Support natif du doigté via `<technical><fingering>`
- Exportable vers tous les logiciels de notation

### 8.3. Feuille de route technologique

| Phase | Approche | Technologie |
|-------|----------|-------------|
| Phase 1 (MVP) | DP classique | PyGuitarPro + NumPy |
| Phase 2 | DP + apprentissage de coût | + scikit-learn |
| Phase 3 | Modèle de séquence | + PyTorch (LSTM/Transformer) |
| Phase 4 | LLM-assisted | + API GPT ou modèle local |

### 8.4. Avantages compétitifs du projet

1. **Focus sur le doigté** : La plupart des outils existants font de la transcription (audio → tab) mais n'annotent pas le doigté
2. **Pipeline modulaire** : Parsers et exporteurs interchangeables
3. **Format Guitar Pro** : Aucun outil open-source n'insère automatiquement le doigté dans les fichiers .gp5
4. **Approche progressive** : Du DP simple vers le ML/DL en fonction des besoins

---

## 9. Références bibliographiques

### Articles fondamentaux

1. **Sayegh, S. I.** (1989). *"Fingering for String Instruments with the Optimum Path Paradigm"*. Computer Music Journal, 13(3), pp. 76-84.

2. **Radicioni, D. P. & Lombardo, V.** (2004). *"Computational Modeling of Guitar Fingering"*. Proceedings of the Joint Conference on AI.

3. **Radisavljevic, A. & Driessen, P.** (2004). *"Path Difference Learning for Guitar Fingering Problem"*. Proceedings of the International Computer Music Conference (ICMC).

4. **Itoh, K. & Hayashida, S.** (2004). *"Optimization for Guitar Fingering on Single Notes"*. Journal of the Institute of Electrical Engineers of Japan, 124(7).

### Recherche récente

5. **Grozman, A. & Norman, J.** (2013). *"An Algorithm for Optimal Guitar Fingering"*. KTH Royal Institute of Technology.

6. **Hori, T. & Sagayama, S.** (2016). *"Minimax Viterbi Algorithm for HMM-based Guitar Fingering Decision"*. Proceedings of ISMIR 2016.

7. **Bontempi, A. et al.** (2024). *"From MIDI to Rich Tablatures: an Automatic Generative System incorporating Lead Guitarists' Fingering and Stylistic Choices"*. Sound and Music Computing Conference (SMC 2024). arXiv:2407.09052.

8. **Kuo, Y. et al.** (2025). *"A Novel Strategy for Difficulty Estimation in the Guitar Fingering Problem"*. Springer.

### Transcription et Deep Learning

9. **Kim, J. & Hayashi, T. et al.** (2022). *"Note-level Automatic Guitar Transcription Using Attention Mechanism"*. EURASIP/EUSIPCO 2022.

10. **Gupta, A. et al.** (2025). *"TART: A Comprehensive Tool for Technique-Aware Audio-to-Tab Guitar Transcription"*. UC Berkeley. arXiv:2510.02597.

11. **Sarmento, P.** (2024). *"Guitar Tablature Generation with Deep Learning"*. PhD Thesis, EURASIP.

### LLM et musique

12. **ChatMusician** (2024). *"ChatMusician: Understanding and Generating Music Intrinsically with LLM"*. ACL Findings 2024.

13. **Midi-LLM** (2025). *"Midi-LLM: Adapting Large Language Models for Text-to-MIDI Music Generation"*. MIT. arXiv:2511.03942.

### Ressources en ligne

14. **awesome-agt** — Collection de ressources pour la transcription automatique de guitare. [GitHub](https://github.com/lucasgris/awesome-agt)

15. **guitar_dp** — Implémentation DP pour le doigté de guitare. [GitHub](https://github.com/jgollub1/guitar_dp)

16. **PyGuitarPro** — Bibliothèque Python pour Guitar Pro. [GitHub](https://github.com/Perlence/PyGuitarPro) | [Documentation](https://pyguitarpro.readthedocs.io/)

17. **music21** — Toolkit d'analyse musicale. [GitHub](https://github.com/cuthbertLab/music21) | [Documentation](https://music21.org/)

18. **pretty-midi** — Utilitaires MIDI. [GitHub](https://github.com/craffel/pretty-midi)
