import json
from pathlib import Path
import tempfile

import importlib

mod = importlib.import_module('cli.assemble')


def test_natural_sort_key():
    assert mod._natural_clip_sort_key('a_edited_001.mp4')[0] == 1
    assert mod._natural_clip_sort_key('a_edited_010.mp4')[0] == 10
    # Fallback when no index
    k = mod._natural_clip_sort_key('a.mp4')[0]
    assert k != 1 and k != 10


def test_parse_criterion_types():
    k, v = mod._parse_criterion('flag:true')
    assert (k, v) == ('flag', True)
    k, v = mod._parse_criterion('num:12')
    assert (k, v) == ('num', 12)
    k, v = mod._parse_criterion('ratio:1.5')
    assert (k, v) == ('ratio', 1.5)
    k, v = mod._parse_criterion('name:Alice')
    assert (k, v) == ('name', 'Alice')


def test_json_get_and_match_criteria(tmp_path: Path):
    data = {
        'age': 'old',
        'describe': {
            'voted': {
                'nsfw': True,
                'score': 0.9,
            }
        }
    }
    p = tmp_path / 'clip.json'
    p.write_text(json.dumps(data), encoding='utf-8')

    # Dot path get
    assert mod._json_get(data, 'describe.voted.nsfw') is True
    assert mod._json_get(data, 'describe.voted.score') == 0.9
    assert mod._json_get(data, 'describe.missing') is None

    # Matching
    assert mod._matches_criteria(str(p), ['age:old']) is True
    assert mod._matches_criteria(str(p), ['describe.voted.nsfw:true']) is True
    assert mod._matches_criteria(str(p), ['describe.voted.nsfw:false']) is False
    assert mod._matches_criteria(str(p), ['age:young']) is False
