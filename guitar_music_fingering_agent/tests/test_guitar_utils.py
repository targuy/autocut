"""Tests pour les utilitaires guitare."""

import pytest
from guitar_fingering.utils.guitar import (
    midi_to_note_name,
    note_to_fret_positions,
    fret_to_midi,
    assign_finger_for_fret,
    STANDARD_TUNING,
)


class TestMidiToNoteName:
    def test_middle_c(self):
        assert midi_to_note_name(60) == 'C4'

    def test_high_e(self):
        assert midi_to_note_name(64) == 'E4'

    def test_low_e(self):
        assert midi_to_note_name(40) == 'E2'

    def test_a440(self):
        assert midi_to_note_name(69) == 'A4'

    def test_invalid_pitch(self):
        with pytest.raises(ValueError):
            midi_to_note_name(128)


class TestNoteToFretPositions:
    def test_open_high_e(self):
        # E4 (64) = corde 1 case 0 en accordage standard
        positions = note_to_fret_positions(64)
        assert (1, 0) in positions

    def test_open_low_e(self):
        # E2 (40) = corde 6 case 0
        positions = note_to_fret_positions(40)
        assert (6, 0) in positions

    def test_multiple_positions(self):
        # A2 (45) = corde 5 case 0, ou corde 6 case 5
        positions = note_to_fret_positions(45)
        assert (5, 0) in positions
        assert (6, 5) in positions

    def test_out_of_range(self):
        # Note trop grave pour la guitare
        positions = note_to_fret_positions(20)
        assert positions == []

    def test_custom_tuning(self):
        # Drop D : corde 6 = D2 (38) au lieu de E2 (40)
        drop_d = [64, 59, 55, 50, 45, 38]
        positions = note_to_fret_positions(38, tuning=drop_d)
        assert (6, 0) in positions


class TestFretToMidi:
    def test_open_string_1(self):
        assert fret_to_midi(1, 0) == 64  # E4

    def test_open_string_6(self):
        assert fret_to_midi(6, 0) == 40  # E2

    def test_fretted_note(self):
        assert fret_to_midi(6, 5) == 45  # A2

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            fret_to_midi(7, 0)


class TestAssignFingerForFret:
    def test_open_string(self):
        assert assign_finger_for_fret(0) == 0

    def test_first_fret(self):
        assert assign_finger_for_fret(1) == 1

    def test_second_fret(self):
        assert assign_finger_for_fret(2) == 2

    def test_high_fret(self):
        # Le doigt ne dépasse jamais 4
        result = assign_finger_for_fret(10)
        assert 1 <= result <= 4

    def test_with_base_position(self):
        # Si la position de base est la case 5, case 5 = doigt 1
        assert assign_finger_for_fret(5, base_position=5) == 1
        assert assign_finger_for_fret(6, base_position=5) == 2
