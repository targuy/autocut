"""
Parser pour les fichiers MusicXML (.xml, .musicxml).

Utilise music21 pour lire les fichiers MusicXML et extraire
les informations de notation, y compris les doigtés existants.
"""

from guitar_fingering.parsers.base import BaseParser
from guitar_fingering.models.music import Track, Measure, Chord, GuitarNote
from guitar_fingering.utils.guitar import note_to_fret_positions

try:
    import music21
except ImportError:
    music21 = None


class MusicXMLParser(BaseParser):
    """Parser pour les fichiers MusicXML (.xml/.musicxml).

    Nécessite la bibliothèque `music21` installée.
    MusicXML est le format qui supporte le mieux les annotations
    de doigté via les éléments <technical><fingering>.
    """

    def supported_extensions(self) -> list[str]:
        return ['.xml', '.musicxml', '.mxl']

    def parse(self, filepath: str) -> Track:
        """Parse un fichier MusicXML.

        Args:
            filepath: Chemin vers le fichier .xml/.musicxml/.mxl.

        Returns:
            Objet Track avec les notes et doigtés existants extraits.

        Raises:
            ImportError: Si music21 n'est pas installé.
        """
        if music21 is None:
            raise ImportError(
                'music21 est requis pour lire les fichiers MusicXML. '
                'Installez-le avec : pip install music21'
            )

        score = music21.converter.parse(filepath)
        track = Track(name='MusicXML Track')

        for part in score.parts:
            for measure_element in part.getElementsByClass('Measure'):
                measure = Measure(number=measure_element.number)

                for note_element in measure_element.notes:
                    if note_element.isNote:
                        midi_pitch = note_element.pitch.midi
                        positions = note_to_fret_positions(midi_pitch)

                        # Chercher un doigté existant dans les articulations
                        finger = None
                        string = None
                        fret = None
                        for art in note_element.articulations:
                            if hasattr(art, 'fingerNumber'):
                                finger = art.fingerNumber
                            if hasattr(art, 'number') and isinstance(
                                art, music21.articulations.StringNumber
                            ):
                                string = art.number
                            if hasattr(art, 'number') and isinstance(
                                art, music21.articulations.Fret
                            ):
                                fret = art.number

                        if string is None or fret is None:
                            if positions:
                                string, fret = positions[-1]
                            else:
                                continue

                        guitar_note = GuitarNote(
                            midi_pitch=midi_pitch,
                            string=string,
                            fret=fret,
                            finger=finger,
                        )
                        chord = Chord(notes=[guitar_note])
                        measure.chords.append(chord)

                track.measures.append(measure)
            break  # Première partie seulement

        return track
