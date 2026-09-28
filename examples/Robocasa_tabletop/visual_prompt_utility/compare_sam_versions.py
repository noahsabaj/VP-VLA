# Compare two running SAM servers (e.g. SAM3 and SAM3.1) on the same frames.
#
# Start one server per version, then:
#   python compare_sam_versions.py --ports 10094 10095 --prompts carrot plate \
#       --inputs /path/to/rollout.mp4 [more.mp4 | frame.png ...] --out sam_compare/
#
# For each sampled frame and prompt it saves a side-by-side picture (top mask of
# each server in red, score and time in the title bar) and writes summary.csv with
# per-call score, mask size and latency. Use raw rollout videos, not overlay videos,
# which already have prompts drawn on them.

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from sam3_client import SAM3Client


def load_frames(inputs, per_video):
    frames = []
    for path in inputs:
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            frames.append((f"{path.stem}", cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)))
            continue
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(0, max(n - 1, 0), per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok:
                frames.append((f"{path.stem}_f{idx:03d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        cap.release()
    return frames


def panel(client, image, masks, label):
    out = client.overlay_masks(image, masks)
    cv2.putText(out, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare SAM servers on the same frames")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", type=int, nargs="+", required=True, help="one port per server to compare")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="videos (.mp4) or frames (.png/.jpg)")
    ap.add_argument("--frames-per-video", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("sam_compare"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    clients = [SAM3Client(host=args.host, port=p) for p in args.ports]
    names = [f"{c._server_metadata.get('model', 'sam')}:{p}" for c, p in zip(clients, args.ports, strict=True)]
    frames = load_frames(args.inputs, args.frames_per_video)

    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "prompt", "server", "num_masks", "top_score", "top_mask_px", "ms"])
        for frame_name, image in frames:
            for prompt in args.prompts:
                panels = []
                for client, name in zip(clients, names, strict=True):
                    t0 = time.perf_counter()
                    res = client.segment(image, prompt, threshold=args.threshold)
                    ms = 1000 * (time.perf_counter() - t0)
                    top_mask, score = np.zeros((0, *image.shape[:2]), np.uint8), 0.0
                    if res["num_masks"]:
                        top = int(np.argmax(res["scores"]))
                        top_mask, score = res["masks"][top : top + 1], float(res["scores"][top])
                    px = int(top_mask.sum())
                    writer.writerow([frame_name, prompt, name, res["num_masks"], f"{score:.3f}", px, f"{ms:.0f}"])
                    panels.append(panel(client, image, top_mask, f"{name} s={score:.2f} {ms:.0f}ms"))
                side = cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(args.out / f"{frame_name}_{prompt.replace(' ', '_')}.png"), side)
    for c in clients:
        c.close()
    print(f"Wrote {len(frames) * len(args.prompts)} comparisons to {args.out}")


if __name__ == "__main__":
    main()
