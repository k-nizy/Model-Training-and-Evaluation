#!/usr/bin/env python3
"""Verify the demo-video clips against the shot list in reports/video_script.md.

What it does:
  1. ffprobe every clips/clip*.mp4 -> duration, and compares it to the per-clip
     time budget from the script (budgets are parsed straight out of the script,
     so the two files cannot drift apart).
  2. Optionally transcribes speech with Faster Whisper and estimates words per
     minute per clip, so you know if the narration will fit when you re-record.
  3. --join concatenates all clips in order into clips/demo_final.mp4 with
     ffmpeg (re-encode concat for safety across different input codecs).

Usage:
  python scripts/video_check.py            # durations + budgets report
  python scripts/video_check.py --voice    # + Whisper transcription / wpm
  python scripts/video_check.py --join     # + build clips/demo_final.mp4
  python scripts/video_check.py --model tiny --lang en --voice --join

Dependencies: ffmpeg/ffprobe on PATH; faster-whisper only for --voice
  (pip install faster-whisper). No whisper install needed otherwise.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_MD = ROOT / "reports" / "video_script.md"
CLIPS_DIR = ROOT / "clips"
OUT_DIR = ROOT / "reports"
TOTAL_BUDGET = 300.0  # 5:00 hard cap


def need_tool(name: str):
    if shutil.which(name) is None:
        sys.exit(f"error: '{name}' not found on PATH. "
                 f"Install ffmpeg (which provides {name}) and retry.")


# ---------------------------------------------------------------- budgets
def parse_budgets(md_text: str):
    """Pull (clip_n, label, budget_seconds) from '## Clip N — ...(...budget Xs,' headings."""
    budgets = {}
    for m in re.finditer(
            r"^## Clip (\d+) — (.+?)\((\d+):(\d+)[–-](\d+):(\d+), budget (\d+)s",
            md_text, re.M):
        n = int(m.group(1))
        start = int(m.group(3)) * 60 + int(m.group(4))
        end = int(m.group(5)) * 60 + int(m.group(6))
        budgets[n] = {
            "title": m.group(2).strip(),
            "start": start, "end": end,
            "budget": float(m.group(7)),
        }
    return budgets


def parse_word_targets(md_text: str):
    """'budget 35s, ~80 words' narration targets per clip, for wpm checking."""
    targets = {}
    text = md_text[: md_text.find("## Shot-list summary")] if "## Shot-list summary" in md_text else md_text
    blocks = re.split(r"^## Clip ", text, flags=re.M)[1:]
    for i, b in enumerate(blocks, 1):
        m = re.search(r"budget (\d+)s, ~(\d+) words", b)
        if m:
            targets[i] = (float(m.group(1)), int(m.group(2)))
    return targets


# ---------------------------------------------------------------- ffprobe
def clip_files():
    files = sorted(CLIPS_DIR.glob("clip*.mp4"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return files


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


# ---------------------------------------------------------------- whisper
def decode_track(path: Path) -> "numpy.ndarray":
    """Extract audio as float32 mono 16 kHz via ffmpeg CLI.

    Skips faster-whisper's internal PyAV decoder, which breaks on some
    installed PyAV versions (e.g. av 19 removed `metadata_errors`).
    ffmpeg is a hard dependency of this script anyway.
    """
    import numpy as np
    import tempfile
    fifo = tempfile.NamedTemporaryFile(suffix=".f32", delete=False)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", "16000", "-f", "f32le", fifo.name]
    subprocess.run(cmd, check=True)
    pcm = Path(fifo.name).read_bytes()
    fifo.close()
    Path(fifo.name).unlink()
    audio = np.frombuffer(pcm, dtype=np.float32)
    if audio.size == 0:
        audio = np.zeros(16000, dtype=np.float32)  # 1s of silence
    return audio


def transcribe(path: Path, model_name: str, lang: str):
    from faster_whisper import WhisperModel  # lazy import; only needed with --voice
    audio = decode_track(path)
    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, info = model.transcribe(audio, language=lang or None, vad_filter=True)
    words, first, last = 0, None, None
    parts = []
    for s in segments:
        parts.append(s.text.strip())
        if first is None:
            first = s.start
        last = s.end
        words += len(s.text.split())
    return " ".join(parts), words, (last - first) if (first is not None and last is not None) else 0.0


# ---------------------------------------------------------------- join
def join(files, out_path: Path):
    list_file = CLIPS_DIR / "_concat.txt"
    with list_file.open("w") as f:
        for p in files:
            f.write(f"file '{p.as_posix()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-c:a", "aac", "-b:a", "128k", str(out_path)],
        check=True)
    list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="store_true",
                    help="transcribe clips with Faster Whisper and report wpm")
    ap.add_argument("--join", action="store_true",
                    help="concatenate clips into clips/demo_final.mp4")
    ap.add_argument("--model", default="base",
                    help="whisper model size: tiny/base/small (default base)")
    ap.add_argument("--lang", default="en", help="spoken language code")
    args = ap.parse_args()

    md = SCRIPT_MD.read_text(encoding="utf-8")
    budgets = parse_budgets(md)
    targets = parse_word_targets(md)
    files = clip_files()

    if not files:
        print(f"No clips found in {CLIPS_DIR}/ (expected clip1.mp4 ... clip{max(budgets)}.mp4)")
        print("Record each clip as its own file, then re-run:")
        print("  python scripts/video_check.py --voice --join")
        return 0

    print(f"{'clip':>5} {'file':>12} {'have':>6} {'budget':>7} {'over':>6}  title")
    total = 0.0
    worst = []

    for n, b in sorted(budgets.items()):
        # match clips by the number embedded in the filename (clip3, clip03, clip_3...)
        match = next((f for f in files if re.fullmatch(rf"clip0?{n}", f.stem)), None)
        if match is None:
            print(f"{n:>5} {'—':>12} {'—':>6} {b['budget']:>6.0f}s {'—':>6}  MISSING: {b['title']}")
            worst.append(n)
            continue
        dur = duration_of(match)
        total += dur
        over = dur - b["budget"]
        flag = "" if over <= 0 else f"  (+{over:.1f}s OVER)"
        print(f"{n:>5} {match.name:>12} {dur:>5.1f}s {b['budget']:>6.0f}s {'':>6}{flag}  {b['title']}")
        if over > 0:
            worst.append(n)

        if args.voice:
            text, words, speech = transcribe(match, args.model, args.lang)
            tgt_words = targets.get(n, (b["budget"], 0))[1]
            wpm = words / (speech / 60) if speech > 1 else 0.0
            vs_budget = f"{wpm:.0f} wpm vs {tgt_words / (b['budget'] / 60):.0f} target"
            if tgt_words and words > tgt_words * 1.15:
                vs_budget += f"  ({words} words vs ~{tgt_words}-word script: TRIM)"
            out_md = OUT_DIR / f"transcript_clip{n}.md"
            out_md.write_text(f"# Clip {n} transcript ({match.name})\n\n{text}\n", encoding="utf-8")
            print(f"       {vs_budget}; transcript -> {out_md.name}")

    missing = [n for n in sorted(budgets) if n in worst and n not in
               {int(re.fullmatch(r"clip0?(\d+)", f.stem).group(1)) for f in files}]
    print(f"\nTotal: {total:.1f}s of {TOTAL_BUDGET:.0f}s budget"
          f"  ({'OK' if total <= TOTAL_BUDGET else f'{total - TOTAL_BUDGET:.1f}s OVER'})")
    if worst:
        print("Clips to re-record or trim:", ", ".join(f"#{n}" for n in sorted(set(worst))))
    else:
        print("All clips within budget.")

    if args.join:
        order = sorted(files, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
        out = OUT_DIR / "demo_final.mp4"
        join(order, out)
        print(f"\nWrote {out} ({duration_of(out):.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
