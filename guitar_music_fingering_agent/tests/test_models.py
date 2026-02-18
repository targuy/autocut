"""Tests pour les modèles de données musicales."""

import pytest
from guitar_fingering.models.music import Note, GuitarNote, Chord, Measure, Track


class TestNote:
    def test_create_valid_note(self):
        n = Note(midi_pitch=64)
        assert n.midi_pitch == 64
        assert n.duration == 0.25
        assert n.velocity == 80

    def test_invalid_midi_pitch_high(self):
        with pytest.raises(ValueError, match='midi_pitch'):
            Note(midi_pitch=128)

    def test_invalid_midi_pitch_low(self):
        with pytest.raises(ValueError, match='midi_pitch'):
            Note(midi_pitch=-1)

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match='duration'):
            Note(midi_pitch=60, duration=-1.0)


class TestGuitarNote:
    def test_create_valid_guitar_note(self):
        gn = GuitarNote(midi_pitch=64, string=1, fret=0)
        assert gn.string == 1
        assert gn.fret == 0
        assert gn.finger is None

    def test_open_string(self):
        gn = GuitarNote(midi_pitch=64, string=1, fret=0, finger=0)
        assert gn.is_open_string is True
        assert gn.has_fingering is True

    def test_fretted_note(self):
        gn = GuitarNote(midi_pitch=65, string=1, fret=1, finger=1)
        assert gn.is_open_string is False
        assert gn.has_fingering is True

    def test_no_fingering(self):
        gn = GuitarNote(midi_pitch=64, string=1, fret=0)
        assert gn.has_fingering is False

    def test_invalid_string(self):
        with pytest.raises(ValueError, match='string'):
            GuitarNote(midi_pitch=64, string=0, fret=0)

    def test_invalid_fret(self):
        with pytest.raises(ValueError, match='fret'):
            GuitarNote(midi_pitch=64, string=1, fret=25)

    def test_invalid_finger(self):
        with pytest.raises(ValueError, match='finger'):
            GuitarNote(midi_pitch=64, string=1, fret=1, finger=5)


class TestChord:
    def test_empty_chord(self):
        c = Chord()
        assert c.size == 0
        assert c.has_complete_fingering is True  # vacuously true

    def test_chord_with_notes(self):
        notes = [
            GuitarNote(midi_pitch=64, string=1, fret=0, finger=0),
            GuitarNote(midi_pitch=59, string=2, fret=0, finger=0),
        ]
        c = Chord(notes=notes)
        assert c.size == 2
        assert c.has_complete_fingering is True

    def test_chord_incomplete_fingering(self):
        notes = [
            GuitarNote(midi_pitch=64, string=1, fret=0, finger=0),
            GuitarNote(midi_pitch=59, string=2, fret=0),
        ]
        c = Chord(notes=notes)
        assert c.has_complete_fingering is False


class TestTrack:
    def test_default_tuning(self):
        t = Track()
        assert t.tuning == [64, 59, 55, 50, 45, 40]
        assert t.capo == 0
        assert t.tempo == 120.0

    def test_total_notes_empty(self):
        t = Track()
        assert t.total_notes == 0

    def test_total_notes_with_data(self):
        notes = [GuitarNote(midi_pitch=64, string=1, fret=0)]
        chord = Chord(notes=notes)
        measure = Measure(number=1, chords=[chord])
        t = Track(measures=[measure])
        assert t.total_notes == 1

    def test_get_all_guitar_notes(self):
        n1 = GuitarNote(midi_pitch=64, string=1, fret=0)
        n2 = GuitarNote(midi_pitch=59, string=2, fret=0)
        chord = Chord(notes=[n1, n2])
        measure = Measure(number=1, chords=[chord])
        t = Track(measures=[measure])
        all_notes = t.get_all_guitar_notes()
        assert len(all_notes) == 2
        assert all_notes[0].midi_pitch == 64
        assert all_notes[1].midi_pitch == 59
