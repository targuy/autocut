"""
Exporteur vers MusicXML avec annotations de doigté.

Utilise music21 pour générer des fichiers MusicXML contenant
les éléments <technical><fingering> pour chaque note.
"""

from guitar_fingering.exporters.base import BaseExporter
from guitar_fingering.models.music import Track

try:
    import music21
except ImportError:
    music21 = None


class MusicXMLExporter(BaseExporter):
    """Exporteur vers fichiers MusicXML (.xml/.musicxml).

    Nécessite la bibliothèque `music21` installée.
    MusicXML supporte nativement les annotations de doigté via
    les éléments <notations><technical>.
    """

    def supported_extensions(self) -> list[str]:
        return ['.xml', '.musicxml']

    def export(self, track: Track, filepath: str) -> None:
        """Exporte une piste avec doigté vers un fichier MusicXML.

        Args:
            track: Objet Track avec doigté assigné.
            filepath: Chemin du fichier .xml/.musicxml de sortie.

        Raises:
            ImportError: Si music21 n'est pas installé.
        """
        if music21 is None:
            raise ImportError(
                'music21 est requis pour exporter vers MusicXML. '
                'Installez-le avec : pip install music21'
            )

        score = music21.stream.Score()
        part = music21.stream.Part()

        for measure_data in track.measures:
            m21_measure = music21.stream.Measure(number=measure_data.number)

            for chord_data in measure_data.chords:
                for note_data in chord_data.notes:
                    n = music21.note.Note(note_data.midi_pitch)
                    n.quarterLength = chord_data.duration

                    # Ajouter le numéro de corde
                    n.articulations.append(
                        music21.articulations.StringNumber(note_data.string)
                    )

                    # Ajouter le doigté si défini
                    if note_data.finger is not None:
                        n.articulations.append(
                            music21.articulations.Fingering(note_data.finger)
                        )

                    m21_measure.append(n)

            part.append(m21_measure)

        score.append(part)
        score.write('musicxml', fp=filepath)
