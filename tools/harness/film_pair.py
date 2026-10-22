"""TWO FILMS SIDE BY SIDE (P3, 2026-10-21: 'take them out and observe'): a student's test film beside its opponent's on the same seed
(tools/harness/demo_film.py's film.mp4 and film_captions.jsonl in each), each captioned with the task under way (its family, toy and phase),
her word and the score so far, and labelled; ffmpeg's hstack. The two tests ran on the same seed, so the tasks come in the same order at
the same ticks until the two bodies' answers change the generator's course (a task ends when it is done or times out): the films are
aligned at the start and drift as the lives do.

    python3 tools/harness/film_pair.py --left DIR --right DIR --out pair.mp4 [--labels "student5 (ours)" "diffusion policy"]"""
import argparse
import json
import os
import subprocess
import sys

FONT = next((f_ for f_ in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                           "/Library/Fonts/Arial.ttf") if os.path.exists(f_)), None)


def fps_of(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    n_, d_ = (out.split("/") + ["1"])[:2]
    return float(n_) / float(d_ or 1)


def srt(dirpath, fps):
    """the captions as subtitles: each from its frame to the next caption's"""
    caps = [json.loads(l_) for l_ in open(os.path.join(dirpath, "film_captions.jsonl")) if l_.strip()]
    def ts(fr):
        s_ = fr / fps; h_ = int(s_ // 3600); m_ = int(s_ % 3600 // 60); sec = s_ % 60
        return f"{h_:02d}:{m_:02d}:{sec:06.3f}".replace(".", ",")
    lines = []
    for i_, c_ in enumerate(caps):
        end = caps[i_ + 1]["frame"] if i_ + 1 < len(caps) else c_["frame"] + int(600 * fps)
        task = c_.get("task"); word = c_.get("word")
        txt = ("no task" if not task else f"{task[0]} ({task[1]}) {task[2] or ''}".strip()) + (f"  |  her word: {word}" if word else "")
        sc = c_.get("score") or {}
        txt += "  |  " + ", ".join(f"{k_} {v_[0]}/{v_[1]}" for k_, v_ in sc.items())
        lines.append(f"{i_ + 1}\n{ts(c_['frame'])} --> {ts(end)}\n{txt}\n")
    p_ = os.path.join(dirpath, "film.srt")
    with open(p_, "w") as f_:
        f_.write("\n".join(lines))
    return p_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True); ap.add_argument("--right", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs=2, default=["student (ours)", "diffusion policy"])
    args = ap.parse_args()
    fl, fr = os.path.join(args.left, "film.mp4"), os.path.join(args.right, "film.mp4")
    fps = fps_of(fl)
    sl, sr = srt(args.left, fps), srt(args.right, fps_of(fr))
    def side(i_, s_, label):
        esc = lambda t_: t_.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\\\\\'")
        font = f":fontfile={FONT}" if FONT else ""
        return (f"[{i_}:v]subtitles={esc(s_)}:force_style='FontSize=16,Outline=1'," 
                f"drawtext=text='{esc(label)}'{font}:fontsize=22:fontcolor=white:borderw=2:x=12:y=12[v{i_}]")
    fc = side(0, sl, args.labels[0]) + ";" + side(1, sr, args.labels[1]) + ";[v0][v1]hstack=inputs=2[v]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", fl, "-i", fr, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", args.out]
    rc = subprocess.call(cmd)
    print(("written " if rc == 0 else f"ffmpeg ended {rc}: ") + args.out, flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
