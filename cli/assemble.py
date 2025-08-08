import os
import sys
import argparse
from collections import defaultdict
import subprocess
import re
import ffmpeg
import json
from typing import Dict, Any, List, Tuple


def is_ffmpeg_installed():
    """Check if ffmpeg is installed and callable."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


if not is_ffmpeg_installed():
    print("Error: ffmpeg is not installed. Please install it to use this script.", file=sys.stderr)
    sys.exit(1)


def get_video_duration(filepath: str) -> float | None:
    """Return video duration in seconds using ffmpeg.probe, or None on error."""
    try:
        probe = ffmpeg.probe(filepath)
        return float(probe['format']['duration'])
    except Exception:
        return None


def get_video_files_by_prefix(src_dir: str) -> Dict[Tuple[str, str], List[str]]:
    """
    Group video files by their prefix (_edited_) in src_dir and all subdirectories.

    Returns:
        dict { (prefix, ext): [relative/path/file1, ...] }
    """
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
    files_by_prefix: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    pattern = re.compile(r"^(.*?)(?:_edited_.*)(\.[^.]+)$")
    for dirpath, _, filenames in os.walk(src_dir):
        for fname in filenames:
            match = pattern.match(fname)
            if match:
                prefix, ext = match.groups()
                if ext.lower() in video_exts:
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(full_path, src_dir)
                    files_by_prefix[(prefix, ext)].append(rel_path)
    return files_by_prefix


def _natural_clip_sort_key(path: str) -> Tuple[int | float, str]:
    """Sort key: extract numeric index after _edited_, fallback to name."""
    name = os.path.basename(path)
    m = re.search(r"_edited_(\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (float('inf'), name)


def _json_get(data: Dict[str, Any], dotted_key: str) -> Any:
    """Retrieve nested value from dict using dot notation (e.g., a.b.c)."""
    cur = data
    for part in dotted_key.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _parse_criterion(expr: str) -> Tuple[str, Any]:
    """Parse 'key:value' into (key, typed_value), typing booleans/numbers when possible."""
    if ':' not in expr:
        return expr.strip(), True  # existence check
    key, val = expr.split(':', 1)
    key = key.strip()
    val = val.strip()
    low = val.lower()
    if low in ('true', 'false'):
        return key, (low == 'true')
    try:
        if '.' in val:
            return key, float(val)
        return key, int(val)
    except ValueError:
        return key, val


def _matches_criteria(json_path: str, criteria: List[str]) -> bool:
    """
    Return True if the JSON sidecar matches all criteria provided as ["k:v", ...].
    - Uses UTF-8 reading.
    - Supports dot-path keys.
    - If JSON missing or invalid, returns False when criteria are provided.
    """
    if not criteria:
        return True
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False
    for expr in criteria:
        key, expected = _parse_criterion(expr)
        actual = _json_get(data, key)
        if isinstance(expected, str) and isinstance(actual, str):
            if actual.strip().lower() != expected.strip().lower():
                return False
        else:
            if actual != expected:
                return False
    return True


def join_videos_by_prefix(src_dir: str, dest_dir: str, files_by_prefix: Dict[Tuple[str, str], List[str]],
                          min_duration: float = 5.0, criteria: List[str] | None = None) -> None:
    """
    Concatenate clips with same prefix, after filtering by duration and optional JSON criteria.
    Always keeps chronological/index order (_edited_### ascending).
    """
    os.makedirs(dest_dir, exist_ok=True)
    for (prefix, ext), files in files_by_prefix.items():
        files_full = [os.path.join(src_dir, f) for f in files]

        # Filter by JSON criteria (if any)
        if criteria:
            filtered = []
            for f in files_full:
                json_sidecar = os.path.splitext(f)[0] + '.json'
                if _matches_criteria(json_sidecar, criteria):
                    filtered.append(f)
            files_full = filtered

        # Filter by min duration
        files_with_duration: List[str] = []
        for f in files_full:
            duration = get_video_duration(f)
            if duration is not None and duration > float(min_duration):
                files_with_duration.append(f)
        if len(files_with_duration) < 2:
            continue  # Only join if more than one valid file

        files_sorted = sorted(files_with_duration, key=_natural_clip_sort_key)
        joined_name = f"{prefix}_joined{ext}"
        joined_path = os.path.join(dest_dir, joined_name)
        filelist_path = os.path.join(dest_dir, f"{prefix}_filelist.txt")
        try:
            with open(filelist_path, 'w', encoding='utf-8') as f:
                for fname in files_sorted:
                    abs_path = os.path.abspath(fname)
                    f.write("file '{}'\n".format(abs_path.replace("'", "''")))
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', filelist_path, '-c', 'copy', joined_path
            ]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: ffmpeg concat failed for prefix '{prefix}': {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error: joining failed for prefix '{prefix}': {e}", file=sys.stderr)
        finally:
            try:
                if os.path.exists(filelist_path):
                    os.remove(filelist_path)
            except Exception:
                pass
        print(f"Joined {files_sorted} into {joined_path}")


def main():
    parser = argparse.ArgumentParser(description="Join video files with the same prefix in a directory.")
    parser.add_argument('src_dir', help="Source directory containing video files")
    parser.add_argument('dest_dir', help="Destination directory for joined videos")
    parser.add_argument('--min-duration', type=float, default=5.0,
                        help="Minimum duration (seconds) for clips to be included in the join (default: 5.0)")
    parser.add_argument('--criteria', action='append', default=[],
                        help="Filter clips by JSON metadata criterion (e.g., --criteria 'age:old' or 'describe.voted.nsfw:true'). Can be used multiple times.")
    args = parser.parse_args()

    files_by_prefix = get_video_files_by_prefix(args.src_dir)
    join_videos_by_prefix(args.src_dir, args.dest_dir, files_by_prefix,
                          min_duration=args.min_duration, criteria=args.criteria)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
