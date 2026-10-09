"""Render foreground segmentation on a video; optional FFmpeg H.264 output."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .checkpoints import load_model, sha256
from .inference import overlay, predict_details
from .models import CLASSES
from .postprocess import validate_config


def run(args):
    if args.output.exists():
        raise ValueError("Output directory already exists")
    if args.batch_size < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("Batch size/max frames must be positive")
    config_path = getattr(args, "postprocess_config", None)
    config = validate_config(json.loads(config_path.read_text())) if config_path else None
    compare = getattr(args, "compare_raw", False)
    if compare and config is None:
        raise ValueError("Comparison requires --postprocess-config")
    model, device, count, ignore_top = load_model(args.model, args.device, args.format, args.ignore_top)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("Cannot open input video")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width < 1 or height < 1 or fps <= 0:
        capture.release()
        raise ValueError("Video has invalid dimensions or frame rate")
    args.output.mkdir(parents=True)
    encoder = None
    writer = None
    stderr_log = None
    processed = 0
    corrections = {}
    try:
        ffmpeg = shutil.which("ffmpeg")
        prefix = "comparison" if compare else "overlay"
        video_path = args.output / (prefix + (".mp4" if ffmpeg else ".avi"))
        rendered_width = width * (2 if compare else 1)
        if ffmpeg:
            stderr_log = (args.output / "ffmpeg.log").open("wb")
            encoder = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{rendered_width}x{height}", "-r", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", str(video_path)
            ], stdin=subprocess.PIPE, stderr=stderr_log)
        else:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                     fps, (rendered_width, height))
            if not writer.isOpened():
                raise RuntimeError("OpenCV video encoder unavailable; install FFmpeg")
        with (args.output / "frames.jsonl").open("w") as stream:
            while args.max_frames is None or processed < args.max_frames:
                frames = []
                limit = args.batch_size if args.max_frames is None else min(
                    args.batch_size, args.max_frames-processed)
                for _ in range(limit):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break
                masks, raw_masks, details = predict_details(model, frames, device, ignore_top, config)
                for frame, mask, raw_mask, detail in zip(frames, masks, raw_masks, details):
                    rendered = overlay(frame, mask)
                    if compare:
                        raw_view = overlay(frame, raw_mask)
                        cv2.putText(raw_view, "RAW", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(rendered, "CORRECTED", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        rendered = np.concatenate((raw_view, rendered), axis=1)
                    for key, value in detail.items():
                        corrections[key] = corrections.get(key, 0) + value
                    if encoder:
                        encoder.stdin.write(rendered.tobytes())
                    else:
                        writer.write(rendered)
                    stream.write(json.dumps({"frame": processed, "seconds": processed/fps,
                        "corrections": detail,
                        "raw_class_pixels": {name: int((raw_mask == i).sum())
                                             for i, name in enumerate(CLASSES[:count])},
                        "class_pixels": {name: int((mask == i).sum())
                                         for i, name in enumerate(CLASSES[:count])}}) + "\n")
                    processed += 1
        if processed == 0:
            raise RuntimeError("Input video contained no decodable frames")
    finally:
        capture.release()
        if writer:
            writer.release()
        if encoder:
            encoder.stdin.close()
            return_code = encoder.wait(timeout=60)
            stderr_log.close()
            if return_code:
                raise RuntimeError("FFmpeg failed; see output/ffmpeg.log")
    report = {"frames": processed, "fps": fps, "classes": list(CLASSES[:count]),
              "ignore_top": ignore_top, "video": video_path.name,
              "model_sha256": sha256(args.model), "audio_included": False,
              "postprocess": config, "postprocess_config_sha256": sha256(config_path) if config_path else None,
              "correction_pixel_totals": corrections, "compare_raw": compare,
              "input_video_sha256": sha256(args.video)}
    (args.output / "prediction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("checkpoint", "torchscript"), default="checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ignore-top", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--postprocess-config", type=Path)
    parser.add_argument("--compare-raw", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
