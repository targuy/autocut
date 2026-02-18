"""
Guitar Music Fingering Agent
Annotation automatique du doigté pour guitare.
"""

__version__ = '0.1.0'

from guitar_fingering.models.music import Note, GuitarNote, Chord, Measure, Track
from guitar_fingering.fingering.engine import FingeringEngine

__all__ = [
    'Note',
    'GuitarNote',
    'Chord',
    'Measure',
    'Track',
    'FingeringEngine',
]
