import os
import sys
import argparse
from collections import defaultdict
import subprocess
import re
import ffmpeg

def is_ffmpeg_installed():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False   
if not is_ffmpeg_installed():
    print("Error: ffmpeg is not installed. Please install it to use this script.", file=sys.stderr)
    sys.exit(1) 

def get_video_duration(filepath):
    try:
        probe = ffmpeg.probe(filepath)
        return float(probe['format']['duration'])
    except Exception:
        return None

def get_video_files_by_prefix(src_dir):
    """
    Groups video files by their prefix (_edited_) in src_dir and all subdirectories.

    Returns:
        dict { (prefix, ext): [file1, file2, ...] }
        - The file paths are relative to src_dir.
    
    How it works:
    - Uses os.walk to recursively traverse src_dir and all its subdirectories.
    - For each file, uses a regex to extract the prefix and extension if it matches the pattern.
    - Only considers files with known video extensions.
    - Stores results in a defaultdict(list) where the key is (prefix, ext) and the value is a list of relative file paths.
    """
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
    files_by_prefix = defaultdict(list)
    # Regex: match prefix + _edited_ + anything + extension
    pattern = re.compile(
        r"^(.*?)(?:_edited_.*)(\.[^.]+)$"
    )
    # os.walk yields (dirpath, dirnames, filenames) for each directory in the tree
    for dirpath, _, filenames in os.walk(src_dir):
        for fname in filenames:
            match = pattern.match(fname)
            if match:
                prefix, ext = match.groups()
                if ext.lower() in video_exts:
                    # Compute the relative path from src_dir to the file
                    full_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(full_path, src_dir)
                    files_by_prefix[(prefix, ext)].append(rel_path)
    return files_by_prefix

def join_videos_by_prefix(src_dir, dest_dir, files_by_prefix):
    os.makedirs(dest_dir, exist_ok=True)
    for (prefix, ext), files in files_by_prefix.items():
        # Filter files by duration > 5 seconds
        files_full = [os.path.join(src_dir, f) for f in files]
        files_with_duration = []
        for f in files_full:
            duration = get_video_duration(f)
            if duration is not None and duration > 5:
                files_with_duration.append(f)
        if len(files_with_duration) < 2:
            continue  # Only join if more than one valid file
        files_sorted = sorted(files_with_duration)
        joined_name = f"{prefix}_joined{ext}"
        joined_path = os.path.join(dest_dir, joined_name)
        filelist_path = os.path.join(dest_dir, f"{prefix}_filelist.txt")
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for fname in files_sorted:
                f.write("file '{}'\n".format(fname.replace("'", "''")))
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', filelist_path, '-c', 'copy', joined_path]
        subprocess.run(cmd, check=True)
        os.remove(filelist_path)
        print(f"Joined {files_sorted} into {joined_path}")

def main():
        parser = argparse.ArgumentParser(description="Join video files with the same prefix in a directory.")
        parser.add_argument('src_dir', help="Source directory containing video files")
        parser.add_argument('dest_dir', help="Destination directory for joined videos")
        args = parser.parse_args()

        files_by_prefix = get_video_files_by_prefix(args.src_dir)
        join_videos_by_prefix(args.src_dir, args.dest_dir, files_by_prefix)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
