#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FROST EMA preprocessing (only FROST).

- Input: .txt/.pos/.csv with columns like ChN_X, ChN_Z (FROST format).
- Output columns (unified):
    time_s, ULx, ULy, LLx, LLy, TTx, TTy, TBx, TBy, TDx, TDy
  where x = horizontal (X), y = vertical (Z).

Pipeline per channel:
  1) Interpolate NaNs (if any)
  2) Low-pass (Butterworth, zero-phase)
  3) Downsample to target_fs (default 50 Hz)
  4) Z-score per channel

Modes:
  * Single file:          --in FILE [--out FILE]
  * Single directory:     --in-dir DIR [--out-dir DIR] [--recurse]
  * Multi-speaker layout: --speakers-root ROOT [--out-root ROOT] (expects <speaker>/pos)

Examples:
  # Single file
  poetry run python ema_preprocess.py --in /path/speaker/pos/file.txt

  # Directory (recursive)
  poetry run python ema_preprocess.py --in-dir /path/speaker/pos --out-dir /path/out --recurse

  # Multi-speaker (<speaker>/pos)
  poetry run python ema_preprocess.py \
      --speakers-root /data/FROST \
      --out-root      /data/FROST \
      --suffix _50Hz
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly

# -----------------------------
# Config
# -----------------------------

TARGET_ORDER = [
    "time_s",
    "ULx","ULy","LLx","LLy",
    "TTx","TTy","TBx","TBy","TDx","TDy"
]

# Default FROST mapping (channel number -> sensor)
# X -> x (horizontal), Z -> y (vertical)
FROST_NUM_TO_SENSOR: Dict[int, str] = {
    12: "UL",
    13: "LL",
    7:  "TT",
    9:  "TB",
    8:  "TD",
}

# Speaker-specific overrides:
# Map sensor -> channel number. When present, it replaces both axes (X and Z).
FROST_SPEAKER_OVERRIDES: Dict[str, Dict[str, int]] = {
    # fin_kh5: TT is on Ch14 instead of Ch7
    "fin_kh5_f": {"TT": 14},
    # rus_kh18: TD is on Ch14 instead of Ch8
    "rus_kh18_m": {"TD": 14},
}

# -----------------------------
# I/O helpers
# -----------------------------

def detect_header(path: Path) -> bool:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                return bool(re.search(r"[A-Za-z_]", s))
    return False

def read_frost_any(path: Path, fs_hint: Optional[float]) -> Tuple[pd.DataFrame, float]:
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        fs = fs_hint if fs_hint is not None else 1250.0
        return df, fs
    if ext in {".pos", ".txt"}:
        has_header = detect_header(path)
        df = pd.read_csv(path, sep=r"\s+", header=0 if has_header else None, engine="python")
        if not has_header:
            df.columns = [f"c{i}" for i in range(df.shape[1])]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        fs = fs_hint if fs_hint is not None else 1250.0
        return df, fs
    raise ValueError(f"Unsupported extension for FROST: {ext} (use .txt/.pos/.csv)")

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def iter_files(root: Path, exts: Iterable[str], recurse: bool):
    exts = {e.lower() for e in exts}
    it = root.rglob("*") if recurse else root.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def suffix_out(path: Path, target_fs: float, out_dir: Optional[Path]=None, root_in: Optional[Path]=None) -> Path:
    suf = f"_{int(target_fs)}Hz.csv"
    if out_dir is None:
        return path.with_name(path.stem + suf)
    assert root_in is not None, "root_in required to preserve relative layout"
    rel = path.relative_to(root_in)
    return (out_dir / rel).with_name(path.stem + suf)

# -----------------------------
# Format check
# -----------------------------

def looks_like_frost(df: pd.DataFrame) -> bool:
    cols = list(map(str, df.columns))
    has_x = any(re.fullmatch(r"Ch\d+_X", c) for c in cols)
    has_z = any(re.fullmatch(r"Ch\d+_Z", c) for c in cols)
    return has_x and has_z

# -----------------------------
# DSP
# -----------------------------

def butter_lowpass_sos(cutoff: float, fs: float, order: int = 6):
    wn = cutoff / (0.5 * fs)
    if not (0 < wn < 1):
        raise ValueError(f"Invalid cutoff {cutoff} for fs={fs}")
    return butter(order, wn, btype="low", output="sos")

def preprocess_matrix(X: np.ndarray, fs: float, target_fs: float, cutoff: float, order: int) -> np.ndarray:
    X = X.astype(float, copy=True)
    # interpolate NaNs
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.any(np.isnan(col)):
            n = len(col)
            idx = np.arange(n)
            mask = ~np.isnan(col)
            X[:, j] = np.interp(idx, idx[mask], col[mask]) if mask.sum() >= 2 else np.nan_to_num(col, nan=0.0)
    # low-pass
    sos = butter_lowpass_sos(cutoff, fs, order)
    Xf = sosfiltfilt(sos, X, axis=0)
    # downsample
    ratio = fs / target_fs
    Xd = resample_poly(Xf, up=1, down=int(round(ratio)), axis=0)
    # z-score
    mean = np.nanmean(Xd, axis=0)
    std = np.nanstd(Xd, axis=0)
    std_safe = np.where((~np.isfinite(std)) | (std == 0), 1.0, std)
    Xz = (Xd - mean) / std_safe
    return Xz

# -----------------------------
# Mapping (FROST)
# -----------------------------

def frost_build_channel_map(speaker: Optional[str]) -> Dict[str, str]:
    """
    Build {"Ch12_X":"ULx", "Ch12_Z":"ULy", ...} using defaults + per-speaker overrides.
    Overrides replace both axes (X and Z) for the specified sensor.
    """
    # default sensor->channel
    sensor_to_chan = {v: k for k, v in FROST_NUM_TO_SENSOR.items()}
    # apply overrides
    if speaker and speaker in FROST_SPEAKER_OVERRIDES:
        for sensor, ch in FROST_SPEAKER_OVERRIDES[speaker].items():
            sensor_to_chan[sensor] = ch
    
    rename_map: Dict[str, str] = {}
    for sensor, ch_num in sensor_to_chan.items():
        rename_map[f"Ch{ch_num}_X"] = f"{sensor}x"
        rename_map[f"Ch{ch_num}_Z"] = f"{sensor}y"
    return rename_map



def frost_keep_and_rename(df: pd.DataFrame, speaker: Optional[str]) -> pd.DataFrame:
    """
    Keep available FROST channels, rename to anatomical schema,
    and apply per-speaker overrides before checking missing columns.
    """
    # Build rename map first — includes overrides (so Ch14 replacement is applied)
    rename_map = frost_build_channel_map(speaker)

    # Now check what’s missing only after the override map is defined
    missing_src = [c for c in rename_map if c not in df.columns]
    if missing_src:
        print(f"[WARN] Missing FROST columns for {speaker or 'file'}: {missing_src}")

    # Keep only columns that actually exist
    keep_src = [c for c in rename_map if c in df.columns]

    # Apply renaming
    out = df[keep_src].rename(columns=rename_map).copy()

    # Fill any missing target columns with NaN
    required = ["ULx","ULy","LLx","LLy","TTx","TTy","TBx","TBy","TDx","TDy"]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan

    # Reorder
    return out[required].copy()

# -----------------------------
# File-level processing
# -----------------------------

def process_dataframe_frost(df_in: pd.DataFrame, fs: float, target_fs: float,
                            cutoff: float, order: int, speaker: Optional[str]) -> pd.DataFrame:
    if not looks_like_frost(df_in):
        raise ValueError("Not FROST: no ChN_X/ChN_Z columns found.")
    df_sel = frost_keep_and_rename(df_in, speaker=speaker)
    X = df_sel.to_numpy(dtype=float)
    Xz = preprocess_matrix(X, fs=fs, target_fs=target_fs, cutoff=cutoff, order=order)
    n = Xz.shape[0]
    time_s = np.arange(n) / target_fs
    out = pd.DataFrame(Xz, columns=df_sel.columns)
    out.insert(0, "time_s", time_s)
    for col in TARGET_ORDER:
        if col not in out.columns:
            out[col] = np.nan
    return out[TARGET_ORDER].copy()

def process_file(in_path: Path, out_path: Path, fs_hint: Optional[float], target_fs: float,
                 cutoff: float, order: int, overwrite: bool, speaker: Optional[str]) -> Tuple[bool, str]:
    try:
        if out_path.exists() and not overwrite:
            return True, f"SKIP (exists): {out_path}"
        df_in, fs = read_frost_any(in_path, fs_hint=fs_hint)
        df_out = process_dataframe_frost(df_in, fs=fs, target_fs=target_fs,
                                         cutoff=cutoff, order=order, speaker=speaker)
        ensure_parent(out_path)
        df_out.to_csv(out_path, index=False)
        return True, f"OK: {in_path.name} -> {out_path.name} (fs {fs} -> {target_fs} Hz)"
    except Exception as e:
        return False, f"ERROR: {in_path} :: {e}"

# -----------------------------
# Batch helpers
# -----------------------------

def list_pos_dirs(speakers_root: Path) -> List[Path]:
    return sorted([d for d in speakers_root.glob("*/pos") if d.is_dir()])

def process_dir(in_dir: Path, out_dir: Optional[Path], fs_hint: Optional[float], target_fs: float,
                cutoff: float, order: int, overwrite: bool, recurse: bool,
                speaker: Optional[str], root_in_for_rel: Optional[Path]=None) -> Tuple[int,int]:
    files = list(iter_files(in_dir, exts=[".csv",".pos",".txt"], recurse=recurse))
    n_ok = 0
    for f in files:
        out_path = suffix_out(f, target_fs=target_fs,
                              out_dir=out_dir, root_in=root_in_for_rel or in_dir)
        ok, msg = process_file(f, out_path, fs_hint, target_fs, cutoff, order, overwrite, speaker)
        print(msg); n_ok += int(ok)
    return n_ok, len(files)

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="FROST EMA preprocessing: low-pass -> resample -> z-score (FROST only).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in", dest="in_file", help="Single file (.txt/.pos/.csv)")
    src.add_argument("--in-dir", dest="in_dir", help="Process all files in this directory")
    src.add_argument("--speakers-root", dest="speakers_root",
                     help="Root with subfolders <speaker>/pos (multi-speaker)")

    ap.add_argument("--out", dest="out_file", help="Output file for --in")
    ap.add_argument("--out-dir", dest="out_dir", help="Output dir for --in-dir")
    ap.add_argument("--out-root", dest="out_root",
                    help="Base output for --speakers-root (default: same as speakers_root)")
    ap.add_argument("--suffix", default="_50Hz", help="Suffix for per-speaker output folders")

    ap.add_argument("--recurse", action="store_true", help="Recurse in --in-dir")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")

    ap.add_argument("--fs", type=float, default=None, help="Input sampling rate (Hz). Default 1250 unless set")
    ap.add_argument("--target-fs", type=float, default=50.0, help="Target sampling rate (Hz)")
    ap.add_argument("--cutoff", type=float, default=20.0, help="Low-pass cutoff (Hz)")
    ap.add_argument("--order", type=int, default=6, help="Butterworth order")

    args = ap.parse_args()

    # Single file
    if args.in_file:
        ip = Path(args.in_file)
        op = Path(args.out_file) if args.out_file else suffix_out(ip, target_fs=args.target_fs)
        speaker = ip.parent.parent.name  
        print(f"[INFO] Detected speaker: {speaker}")
        ok, msg = process_file(ip, op, args.fs, args.target_fs, args.cutoff, args.order,
                               args.overwrite, speaker=speaker)
        
        print(msg); sys.exit(0 if ok else 1)

    # Multi-speaker (<speaker>/pos)
    if args.speakers_root:
        root = Path(args.speakers_root)
        out_root = Path(args.out_root) if args.out_root else root
        pos_dirs = list_pos_dirs(root)
        if not pos_dirs:
            print(f"No '*/pos' subfolders found in: {root}", file=sys.stderr); sys.exit(1)

        total_ok, total_all = 0, 0
        for pos_dir in pos_dirs:
            speaker = pos_dir.parent.name
            speaker_out_dir = out_root / f"renamed_{speaker}{args.suffix}"
            print(f"\n=== Speaker: {speaker} ===")
            n_ok, n_all = process_dir(
                in_dir=pos_dir,
                out_dir=speaker_out_dir,
                fs_hint=args.fs,
                target_fs=args.target_fs,
                cutoff=args.cutoff,
                order=args.order,
                overwrite=args.overwrite,
                recurse=True,
                speaker=speaker,
                root_in_for_rel=pos_dir,
            )
            total_ok += n_ok; total_all += n_all
            print(f"Speaker {speaker}: {n_ok}/{n_all} processed -> {speaker_out_dir}")
        print(f"\nDone. {total_ok}/{total_all} files processed successfully.")
        sys.exit(0)

    # Single directory (single speaker)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    speaker = in_dir.parent.name
    print(f"[INFO] Detected speaker: {speaker}")    
    
    n_ok, n_all = process_dir(
        in_dir=in_dir,
        out_dir=out_dir,
        fs_hint=args.fs,
        target_fs=args.target_fs,
        cutoff=args.cutoff,
        order=args.order,
        overwrite=args.overwrite,
        recurse=args.recurse,
        speaker=speaker,
        root_in_for_rel=in_dir,
    )
    print(f"Done. {n_ok}/{n_all} files processed. Output -> {out_dir if out_dir else 'next to inputs'}")

if __name__ == "__main__":
    main()
