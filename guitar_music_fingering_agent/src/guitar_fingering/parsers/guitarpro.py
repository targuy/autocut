"""
Parser pour les fichiers Guitar Pro (.gp3, .gp4, .gp5).

Utilise la bibliothèque PyGuitarPro pour lire les fichiers Guitar Pro
et extraire les notes, cordes, cases et effets.
"""

from guitar_fingering.parsers.base import BaseParser
from guitar_fingering.models.music import Track, Measure, Chord, GuitarNote

try:
    import guitarpro
except ImportError:
    guitarpro = None


class GuitarProParser(BaseParser):
    """Parser pour les fichiers Guitar Pro (.gp3/.gp4/.gp5).

    Nécessite la bibliothèque `pyguitarpro` installée.
    """

    def supported_extensions(self) -> list[str]:
        return ['.gp3', '.gp4', '.gp5']

    def parse(self, filepath: str, track_index: int = 0) -> Track:
        """Parse un fichier Guitar Pro.

        Args:
            filepath: Chemin vers le fichier .gp3/.gp4/.gp5.
            track_index: Index de la piste à extraire (défaut: 0).

        Returns:
            Objet Track avec les notes et positions extraites.

        Raises:
            ImportError: Si PyGuitarPro n'est pas installé.
            FileNotFoundError: Si le fichier n'existe pas.
        """
        if guitarpro is None:
            raise ImportError(
                'PyGuitarPro est requis pour lire les fichiers Guitar Pro. '
                'Installez-le avec : pip install pyguitarpro'
            )

        song = guitarpro.parse(filepath)
        gp_track = song.tracks[track_index]

        # Extraire l'accordage
        tuning = [s.value for s in gp_track.strings]

        track = Track(
            name=gp_track.name,
            tuning=tuning,
            tempo=song.tempo,
        )

        for measure_idx, gp_measure in enumerate(gp_track.measures):
            measure = Measure(number=measure_idx + 1)

            for voice in gp_measure.voices:
                for beat in voice.beats:
                    chord = Chord(duration=beat.duration.time)
                    for gp_note in beat.notes:
                        midi_pitch = tuning[gp_note.string - 1] + gp_note.value
                        guitar_note = GuitarNote(
                            midi_pitch=midi_pitch,
                            string=gp_note.string,
                            fret=gp_note.value,
                            finger=None,  # Sera calculé par l'optimiseur
                        )
                        chord.notes.append(guitar_note)

                    if chord.notes:
                        measure.chords.append(chord)

            track.measures.append(measure)

        return track
