"""
Moteur principal de doigté pour guitare.

Orchestre le pipeline complet : chargement du fichier, extraction des notes,
optimisation du doigté et export du résultat.
"""

from typing import Optional

from guitar_fingering.models.music import Track, Measure, Chord, GuitarNote
from guitar_fingering.fingering.optimizer import optimize_fingering
from guitar_fingering.utils.guitar import STANDARD_TUNING


class FingeringEngine:
    """Moteur de calcul automatique du doigté pour guitare.

    Usage:
        engine = FingeringEngine()
        engine.load('morceau.gp5')
        engine.compute_fingering()
        engine.export('morceau_avec_doigte.gp5')
    """

    def __init__(self, tuning: Optional[list[int]] = None):
        """Initialise le moteur de doigté.

        Args:
            tuning: Accordage personnalisé (notes MIDI des cordes à vide).
                    Par défaut : accordage standard EADGBE.
        """
        self._tuning = tuning or STANDARD_TUNING
        self._track: Optional[Track] = None

    @property
    def track(self) -> Optional[Track]:
        """Retourne la piste chargée."""
        return self._track

    def load_track(self, track: Track) -> None:
        """Charge une piste guitare directement.

        Args:
            track: Objet Track contenant les mesures et notes.
        """
        self._track = track
        self._tuning = track.tuning

    def compute_fingering(self) -> Track:
        """Calcule le doigté optimal pour toutes les notes de la piste.

        Returns:
            Track avec doigté assigné à chaque note.

        Raises:
            ValueError: Si aucune piste n'est chargée.
        """
        if self._track is None:
            raise ValueError('Aucune piste chargée. Utilisez load_track() d\'abord.')

        # Extraire toutes les notes MIDI de la piste
        all_notes = self._track.get_all_guitar_notes()

        if not all_notes:
            return self._track

        # Extraire les pitches MIDI pour l'optimisation
        midi_pitches = [n.midi_pitch for n in all_notes]

        # Optimiser le doigté
        optimized_notes = optimize_fingering(midi_pitches, self._tuning)

        # Réinjecter les doigtés optimisés dans la piste
        note_idx = 0
        for measure in self._track.measures:
            for chord in measure.chords:
                for i, _note in enumerate(chord.notes):
                    if note_idx < len(optimized_notes):
                        chord.notes[i] = GuitarNote(
                            midi_pitch=optimized_notes[note_idx].midi_pitch,
                            string=optimized_notes[note_idx].string,
                            fret=optimized_notes[note_idx].fret,
                            finger=optimized_notes[note_idx].finger,
                            start_time=_note.start_time,
                            duration=_note.duration,
                        )
                        note_idx += 1

        return self._track
