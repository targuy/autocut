"""
Parser pour les fichiers MIDI (.mid, .midi).

Utilise pretty_midi pour lire les fichiers MIDI et convertir les notes
en positions sur le manche de guitare.
"""

from guitar_fingering.parsers.base import BaseParser
from guitar_fingering.models.music import Track, Measure, Chord, GuitarNote
from guitar_fingering.utils.guitar import note_to_fret_positions

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


class MidiParser(BaseParser):
    """Parser pour les fichiers MIDI (.mid/.midi).

    Nécessite la bibliothèque `pretty-midi` installée.
    Note : Le MIDI ne contient pas d'information de corde/case,
    donc une heuristique est utilisée pour le placement sur le manche.
    """

    def supported_extensions(self) -> list[str]:
        return ['.mid', '.midi']

    def parse(self, filepath: str, instrument_index: int = 0) -> Track:
        """Parse un fichier MIDI.

        Args:
            filepath: Chemin vers le fichier .mid/.midi.
            instrument_index: Index de l'instrument à extraire (défaut: 0).

        Returns:
            Objet Track avec les notes converties en positions guitare.

        Raises:
            ImportError: Si pretty-midi n'est pas installé.
        """
        if pretty_midi is None:
            raise ImportError(
                'pretty-midi est requis pour lire les fichiers MIDI. '
                'Installez-le avec : pip install pretty-midi'
            )

        pm = pretty_midi.PrettyMIDI(filepath)

        if not pm.instruments:
            return Track(name='Empty')

        instrument = pm.instruments[instrument_index]
        track = Track(name=instrument.name or 'Guitar')

        # Grouper les notes par temps de début pour former les accords
        tempo = pm.estimate_tempo()
        track.tempo = tempo

        notes_by_time: dict[float, list] = {}
        for note in instrument.notes:
            # Arrondir le temps pour regrouper les notes simultanées
            t = round(note.start, 3)
            if t not in notes_by_time:
                notes_by_time[t] = []
            notes_by_time[t].append(note)

        measure = Measure(number=1)
        for start_time in sorted(notes_by_time.keys()):
            midi_notes = notes_by_time[start_time]
            chord = Chord(start_time=start_time)

            for midi_note in midi_notes:
                positions = note_to_fret_positions(midi_note.pitch)
                if positions:
                    # Choisir la position la plus basse par défaut
                    string, fret = positions[-1]
                    guitar_note = GuitarNote(
                        midi_pitch=midi_note.pitch,
                        string=string,
                        fret=fret,
                        finger=None,
                        start_time=midi_note.start,
                        duration=midi_note.end - midi_note.start,
                    )
                    chord.notes.append(guitar_note)

            if chord.notes:
                measure.chords.append(chord)

        track.measures.append(measure)
        return track
