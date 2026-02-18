"""
Fonctions de coût ergonomique pour l'optimisation du doigté guitare.

Évalue le coût de transition entre deux positions successives sur le manche,
en prenant en compte des critères biomécaniques et ergonomiques.
"""

from guitar_fingering.models.music import GuitarNote
from guitar_fingering.utils.guitar import MAX_FINGER_SPAN


def transition_cost(note_from: GuitarNote, note_to: GuitarNote) -> float:
    """Calcule le coût de transition entre deux notes successives.

    Prend en compte :
    - La distance de déplacement sur le manche (cases)
    - Le changement de corde
    - L'écart entre les doigts
    - Les pénalités pour positions inconfortables

    Args:
        note_from: Note de départ (avec doigté assigné).
        note_to: Note d'arrivée (avec doigté assigné).

    Returns:
        Coût de la transition (float >= 0). Plus le coût est bas,
        plus la transition est ergonomique.
    """
    cost = 0.0

    # Coût de déplacement sur le manche (distance en cases)
    fret_distance = abs(note_to.fret - note_from.fret)
    cost += fret_distance * 1.0

    # Coût de changement de corde
    string_distance = abs(note_to.string - note_from.string)
    cost += string_distance * 0.5

    # Pénalité si l'écart entre les doigts est trop grand
    if note_from.finger is not None and note_to.finger is not None:
        if note_from.finger > 0 and note_to.finger > 0:
            finger_span = abs(note_to.fret - note_from.fret)
            if finger_span > MAX_FINGER_SPAN:
                cost += (finger_span - MAX_FINGER_SPAN) * 3.0

    # Pénalité pour les positions très hautes sur le manche
    if note_to.fret > 12:
        cost += (note_to.fret - 12) * 0.2

    # Bonus pour les cordes à vide (plus faciles)
    if note_to.fret == 0:
        cost -= 0.5

    return max(0.0, cost)


def position_cost(note: GuitarNote) -> float:
    """Calcule le coût intrinsèque d'une position sur le manche.

    Args:
        note: Note avec sa position (corde, case, doigt).

    Returns:
        Coût de la position (float >= 0).
    """
    cost = 0.0

    # Les cases basses sont généralement plus confortables
    if note.fret > 0:
        cost += note.fret * 0.1

    # Corde à vide = position la plus facile
    if note.fret == 0:
        cost = 0.0

    return cost
