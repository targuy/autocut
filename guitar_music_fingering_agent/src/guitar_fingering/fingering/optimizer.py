"""
Optimiseur de doigté par programmation dynamique (DP).

Implémente l'algorithme de Viterbi / plus court chemin pour trouver
la séquence de doigtés optimale minimisant le coût ergonomique total.
"""

from typing import Optional

from guitar_fingering.models.music import GuitarNote
from guitar_fingering.fingering.cost import transition_cost, position_cost
from guitar_fingering.utils.guitar import (
    note_to_fret_positions,
    assign_finger_for_fret,
)


def generate_candidates(
    midi_pitch: int,
    tuning: Optional[list[int]] = None,
) -> list[GuitarNote]:
    """Génère toutes les positions candidates (corde, case, doigt) pour une note.

    Pour chaque position possible sur le manche, génère les doigtés plausibles.

    Args:
        midi_pitch: Numéro de note MIDI.
        tuning: Accordage (optionnel, défaut = standard).

    Returns:
        Liste de GuitarNote candidates avec doigté assigné.
    """
    candidates = []
    positions = note_to_fret_positions(midi_pitch, tuning)

    for string, fret in positions:
        if fret == 0:
            # Corde à vide : doigt = 0
            candidates.append(GuitarNote(
                midi_pitch=midi_pitch,
                string=string,
                fret=fret,
                finger=0,
            ))
        else:
            # Assigne le doigt le plus naturel selon la position
            finger = assign_finger_for_fret(fret)
            candidates.append(GuitarNote(
                midi_pitch=midi_pitch,
                string=string,
                fret=fret,
                finger=finger,
            ))
    return candidates


def optimize_fingering(
    midi_pitches: list[int],
    tuning: Optional[list[int]] = None,
) -> list[GuitarNote]:
    """Optimise le doigté d'une séquence de notes par programmation dynamique.

    Utilise l'algorithme de Viterbi pour trouver le chemin de coût minimal
    à travers le graphe des positions candidates.

    Args:
        midi_pitches: Séquence de notes MIDI à doigter.
        tuning: Accordage (optionnel, défaut = standard).

    Returns:
        Liste de GuitarNote avec doigté optimal assigné.
    """
    if not midi_pitches:
        return []

    # Générer les candidats pour chaque note
    all_candidates = [generate_candidates(p, tuning) for p in midi_pitches]

    # Filtrer les notes sans candidats (hors tessiture de la guitare)
    valid_indices = [i for i, c in enumerate(all_candidates) if c]
    if not valid_indices:
        return []

    # Programmation dynamique : tables de coûts et de backtracking
    n = len(all_candidates)
    dp_cost = [None] * n       # Coût minimal pour atteindre chaque candidat
    dp_parent = [None] * n     # Index du parent optimal pour backtracking

    # Initialisation : première note
    first_idx = valid_indices[0]
    dp_cost[first_idx] = [position_cost(c) for c in all_candidates[first_idx]]
    dp_parent[first_idx] = [None] * len(all_candidates[first_idx])

    # Propagation
    prev_idx = first_idx
    for vi in range(1, len(valid_indices)):
        curr_idx = valid_indices[vi]
        curr_candidates = all_candidates[curr_idx]
        prev_candidates = all_candidates[prev_idx]

        dp_cost[curr_idx] = [float('inf')] * len(curr_candidates)
        dp_parent[curr_idx] = [0] * len(curr_candidates)

        for j, curr_note in enumerate(curr_candidates):
            for k, prev_note in enumerate(prev_candidates):
                cost = dp_cost[prev_idx][k] + transition_cost(prev_note, curr_note) + position_cost(curr_note)
                if cost < dp_cost[curr_idx][j]:
                    dp_cost[curr_idx][j] = cost
                    dp_parent[curr_idx][j] = k

        prev_idx = curr_idx

    # Backtracking : retrouver le chemin optimal
    last_idx = valid_indices[-1]
    best_j = min(range(len(dp_cost[last_idx])), key=lambda j: dp_cost[last_idx][j])

    result = [None] * n
    j = best_j
    for vi in range(len(valid_indices) - 1, -1, -1):
        idx = valid_indices[vi]
        result[idx] = all_candidates[idx][j]
        if dp_parent[idx] is not None and dp_parent[idx][j] is not None:
            j = dp_parent[idx][j]

    # Retourner uniquement les notes valides
    return [r for r in result if r is not None]
