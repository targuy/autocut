"""
Utilitaires pour la guitare : constantes, accordages et helpers.

Fournit les fonctions de base pour la conversion entre notes MIDI
et positions sur le manche de la guitare.
"""

from typing import Optional

# Accordage standard EADGBE (MIDI notes des cordes à vide)
# Corde 1 (Mi aigu) = 64, Corde 6 (Mi grave) = 40
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]

# Nombre maximal de cases sur le manche
MAX_FRET = 24

# Noms des doigts de la main gauche
FINGER_NAMES = {
    0: 'ouvert',
    1: 'index',
    2: 'majeur',
    3: 'annulaire',
    4: 'auriculaire',
}

# Nombre maximal d'écart entre doigts (en cases) pour une position confortable
MAX_FINGER_SPAN = 5

# Noms des notes
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_note_name(midi_pitch: int) -> str:
    """Convertit un numéro MIDI en nom de note (ex: 64 -> 'E4').

    Args:
        midi_pitch: Numéro de note MIDI (0-127).

    Returns:
        Nom de la note avec octave (ex: 'E4', 'A2').
    """
    if not 0 <= midi_pitch <= 127:
        raise ValueError(f'midi_pitch doit être entre 0 et 127, reçu: {midi_pitch}')
    note_name = NOTE_NAMES[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1
    return f'{note_name}{octave}'


def note_to_fret_positions(
    midi_pitch: int,
    tuning: Optional[list[int]] = None,
    max_fret: int = MAX_FRET,
) -> list[tuple[int, int]]:
    """Trouve toutes les positions (corde, case) possibles pour une note MIDI.

    Args:
        midi_pitch: Numéro de note MIDI.
        tuning: Accordage (liste de notes MIDI des cordes à vide).
                Par défaut : accordage standard EADGBE.
        max_fret: Nombre maximal de cases considérées.

    Returns:
        Liste de tuples (numéro_corde, numéro_case).
        Le numéro de corde va de 1 (aigu) à 6 (grave).
    """
    if tuning is None:
        tuning = STANDARD_TUNING

    positions = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_pitch - open_note
        if 0 <= fret <= max_fret:
            positions.append((string_idx + 1, fret))
    return positions


def fret_to_midi(string: int, fret: int, tuning: Optional[list[int]] = None) -> int:
    """Convertit une position (corde, case) en note MIDI.

    Args:
        string: Numéro de corde (1=aigu à 6=grave).
        fret: Numéro de case (0=corde à vide).
        tuning: Accordage. Par défaut : standard EADGBE.

    Returns:
        Numéro de note MIDI.
    """
    if tuning is None:
        tuning = STANDARD_TUNING

    if not 1 <= string <= len(tuning):
        raise ValueError(f'string doit être entre 1 et {len(tuning)}, reçu: {string}')

    return tuning[string - 1] + fret


def assign_finger_for_fret(fret: int, base_position: int = 1) -> int:
    """Assigne un doigt par défaut pour une case donnée.

    Utilise une heuristique simple : le doigt correspond à l'écart
    par rapport à la position de base (index = case la plus basse).

    Args:
        fret: Numéro de case.
        base_position: Case de référence pour l'index (1er doigt).

    Returns:
        Numéro de doigt (0=ouvert, 1-4).
    """
    if fret == 0:
        return 0  # Corde à vide

    offset = fret - base_position
    finger = max(1, min(4, offset + 1))
    return finger
