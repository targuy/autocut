"""
Exporteur vers Guitar Pro (.gp5).

Génère un fichier Guitar Pro avec les annotations de doigté
intégrées aux notes.
"""

from guitar_fingering.exporters.base import BaseExporter
from guitar_fingering.models.music import Track

try:
    import guitarpro
except ImportError:
    guitarpro = None


class GuitarProExporter(BaseExporter):
    """Exporteur vers fichiers Guitar Pro (.gp5).

    Nécessite la bibliothèque `pyguitarpro` installée.
    """

    def supported_extensions(self) -> list[str]:
        return ['.gp5']

    def export(self, track: Track, filepath: str) -> None:
        """Exporte une piste avec doigté vers un fichier Guitar Pro.

        Args:
            track: Objet Track avec doigté assigné.
            filepath: Chemin du fichier .gp5 de sortie.

        Raises:
            ImportError: Si PyGuitarPro n'est pas installé.
        """
        if guitarpro is None:
            raise ImportError(
                'PyGuitarPro est requis pour exporter vers Guitar Pro. '
                'Installez-le avec : pip install pyguitarpro'
            )

        song = guitarpro.models.Song()
        song.tempo = int(track.tempo)

        gp_track = guitarpro.models.Track()
        gp_track.name = track.name
        gp_track.strings = [
            guitarpro.models.GuitarString(number=i + 1, value=v)
            for i, v in enumerate(track.tuning)
        ]

        for measure_data in track.measures:
            gp_measure = guitarpro.models.Measure(gp_track.measures[-1].header if gp_track.measures else None)
            voice = gp_measure.voices[0]

            for chord_data in measure_data.chords:
                beat = guitarpro.models.Beat(voice)
                for note_data in chord_data.notes:
                    gp_note = guitarpro.models.Note(beat)
                    gp_note.string = note_data.string
                    gp_note.value = note_data.fret
                    if note_data.finger is not None:
                        gp_note.effect.leftHandFinger = guitarpro.models.Fingering(note_data.finger)
                    beat.notes.append(gp_note)
                voice.beats.append(beat)

            gp_track.measures.append(gp_measure)

        song.tracks.append(gp_track)
        guitarpro.write(song, filepath)
