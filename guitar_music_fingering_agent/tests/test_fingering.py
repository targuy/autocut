"""Tests pour le moteur de doigté et l'optimiseur."""

import pytest
from guitar_fingering.models.music import GuitarNote, Chord, Measure, Track
from guitar_fingering.fingering.optimizer import generate_candidates, optimize_fingering
from guitar_fingering.fingering.cost import transition_cost, position_cost
from guitar_fingering.fingering.engine import FingeringEngine


class TestGenerateCandidates:
    def test_open_e_string(self):
        # E4 (64) devrait inclure corde 1 case 0
        candidates = generate_candidates(64)
        assert any(c.string == 1 and c.fret == 0 for c in candidates)

    def test_candidates_have_fingering(self):
        candidates = generate_candidates(64)
        for c in candidates:
            assert c.finger is not None

    def test_open_string_finger_is_zero(self):
        candidates = generate_candidates(64)
        open_strings = [c for c in candidates if c.fret == 0]
        for c in open_strings:
            assert c.finger == 0

    def test_no_candidates_for_out_of_range(self):
        # Note trop grave pour la guitare standard
        candidates = generate_candidates(20)
        assert candidates == []


class TestTransitionCost:
    def test_same_position_zero_cost(self):
        n1 = GuitarNote(midi_pitch=64, string=1, fret=0, finger=0)
        n2 = GuitarNote(midi_pitch=64, string=1, fret=0, finger=0)
        cost = transition_cost(n1, n2)
        assert cost == 0.0

    def test_fret_distance_increases_cost(self):
        n1 = GuitarNote(midi_pitch=64, string=1, fret=0, finger=0)
        n2 = GuitarNote(midi_pitch=69, string=1, fret=5, finger=1)
        n3 = GuitarNote(midi_pitch=76, string=1, fret=12, finger=1)
        cost_near = transition_cost(n1, n2)
        cost_far = transition_cost(n1, n3)
        assert cost_far > cost_near

    def test_open_string_bonus(self):
        # Verify open string bonus reduces cost compared to same note without bonus
        n = GuitarNote(midi_pitch=64, string=1, fret=0, finger=0)
        cost = position_cost(n)
        assert cost == 0.0  # Open string has zero position cost


class TestPositionCost:
    def test_open_string_is_free(self):
        n = GuitarNote(midi_pitch=64, string=1, fret=0, finger=0)
        assert position_cost(n) == 0.0

    def test_higher_fret_costs_more(self):
        n_low = GuitarNote(midi_pitch=65, string=1, fret=1, finger=1)
        n_high = GuitarNote(midi_pitch=76, string=1, fret=12, finger=1)
        assert position_cost(n_high) > position_cost(n_low)


class TestOptimizeFingering:
    def test_empty_sequence(self):
        result = optimize_fingering([])
        assert result == []

    def test_single_note(self):
        result = optimize_fingering([64])
        assert len(result) == 1
        assert result[0].midi_pitch == 64
        assert result[0].finger is not None

    def test_ascending_scale(self):
        # Gamme ascendante E4 -> B4
        pitches = [64, 65, 67, 69, 71]
        result = optimize_fingering(pitches)
        assert len(result) == 5
        # Chaque note doit avoir un doigté assigné
        for note in result:
            assert note.finger is not None

    def test_repeated_note(self):
        result = optimize_fingering([64, 64, 64])
        assert len(result) == 3
        # Toutes les notes devraient avoir la même position
        assert all(r.string == result[0].string for r in result)

    def test_out_of_range_notes_filtered(self):
        result = optimize_fingering([20, 64, 20])
        # Notes hors tessiture sont filtrées
        assert len(result) >= 1


class TestFingeringEngine:
    def test_no_track_raises(self):
        engine = FingeringEngine()
        with pytest.raises(ValueError, match='Aucune piste'):
            engine.compute_fingering()

    def test_compute_with_track(self):
        notes = [
            GuitarNote(midi_pitch=64, string=1, fret=0),
            GuitarNote(midi_pitch=65, string=1, fret=1),
            GuitarNote(midi_pitch=67, string=1, fret=3),
        ]
        chords = [Chord(notes=[n]) for n in notes]
        measure = Measure(number=1, chords=chords)
        track = Track(measures=[measure])

        engine = FingeringEngine()
        engine.load_track(track)
        result = engine.compute_fingering()

        # Toutes les notes doivent avoir un doigté
        all_notes = result.get_all_guitar_notes()
        assert len(all_notes) == 3
        for note in all_notes:
            assert note.has_fingering

    def test_empty_track(self):
        track = Track()
        engine = FingeringEngine()
        engine.load_track(track)
        result = engine.compute_fingering()
        assert result.total_notes == 0
