"""Drivable label review tool: stdlib HTTP server + SAM2.1 point/box prompts on the GPU.

    python server.py --work ~/drivable-review/work [--port 8790] [--host 0.0.0.0]

work/
  index.json        [{id, image, split, flag, note, ...}]  (frame list, from the first-pass review)
  images/<id>.jpg   320x240 RGB
  masks_v1/<id>.png first-pass corrected mask (6 classes + 255)
  masks_orig/<id>.png rosy D-554 mask, for comparison
  saved/<id>.png    your edits (created on save)
  status.json       {id: {"status": "done|hold|exclude", "at": "..."}}
"""
import argparse, base64, io, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--work", type=Path, required=True)
ap.add_argument("--port", type=int, default=8790)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--sam-root", type=Path, default=Path.home() / "pinky-sam-benchmark")
args = ap.parse_args()
W = args.work.expanduser()
STATIC = Path(__file__).parent
INDEX = json.loads((W / "index.json").read_text())
IDS = {e["id"]: e for e in INDEX}
STATUS_FILE = W / "status.json"
STATUS = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
(W / "saved").mkdir(exist_ok=True)
lock = threading.Lock()

# ---- SAM2 (lazy, one image embedding cached) ----
_sam = {"pred": None, "id": None}
sam_lock = threading.Lock()


def sam_predictor():
    if _sam["pred"] is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = args.sam_root / "checkpoints" / "sam2" / "sam2.1_hiera_large.pt"
        model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(ckpt), device="cuda")
        _sam["pred"] = SAM2ImagePredictor(model)
        _sam["torch"] = torch
    return _sam["pred"]


def sam_mask(fid, points, box):
    with sam_lock:
        pred = sam_predictor()
        torch = _sam["torch"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if _sam["id"] != fid:
                img = np.asarray(Image.open(W / "images" / f"{fid}.jpg").convert("RGB"))
                pred.set_image(img)
                _sam["id"] = fid
            kw = {}
            if points:
                kw["point_coords"] = np.array([[p[0], p[1]] for p in points], np.float32)
                kw["point_labels"] = np.array([p[2] for p in points], np.int32)
            if box:
                kw["box"] = np.array(box, np.float32)
            masks, scores, _ = pred.predict(multimask_output=not box and len(points) == 1, **kw)
        best = masks[int(np.argmax(scores))] > 0
        return best.astype(np.uint8)


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def current_mask_path(fid):
    p = W / "saved" / f"{fid}.png"
    return p if p.exists() else W / "masks_v1" / f"{fid}.png"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/list":
            out = []
            for e in INDEX:
                s = STATUS.get(e["id"], {})
                out.append({**e, "status": s.get("status", ""), "saved": (W / "saved" / f"{e['id']}.png").exists()})
            return self.send(200, out)
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[1] in IDS:
            kind, fid = parts
            if kind == "img":
                return self.send(200, (W / "images" / f"{fid}.jpg").read_bytes(), "image/jpeg")
            if kind == "mask":
                return self.send(200, current_mask_path(fid).read_bytes(), "image/png")
            if kind == "orig":
                return self.send(200, (W / "masks_orig" / f"{fid}.png").read_bytes(), "image/png")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if len(parts) != 3 or parts[0] != "api" or parts[2] not in IDS:
            return self.send(404, {"error": "not found"})
        op, fid = parts[1], parts[2]
        if op == "save":
            mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["png"]))))
            if mask.shape != (240, 320) or mask.dtype != np.uint8:
                return self.send(400, {"error": f"bad mask {mask.shape} {mask.dtype}"})
            orig = np.asarray(Image.open(W / "masks_v1" / f"{fid}.png"))
            lane = (orig >= 1) & (orig <= 4)
            if np.any(mask[lane] != orig[lane]):
                return self.send(400, {"error": "lane pixels changed"})
            bad = ~np.isin(mask, [0, 1, 2, 3, 4, 5, 255])
            if bad.any():
                return self.send(400, {"error": "invalid class value"})
            Image.fromarray(mask).save(W / "saved" / f"{fid}.png")
            with lock:
                if body.get("status"):
                    STATUS[fid] = {"status": body["status"], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                STATUS_FILE.write_text(json.dumps(STATUS, indent=1))
            return self.send(200, {"ok": True})
        if op == "status":
            with lock:
                if body.get("status"):
                    STATUS[fid] = {"status": body["status"], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
                else:
                    STATUS.pop(fid, None)
                STATUS_FILE.write_text(json.dumps(STATUS, indent=1))
            return self.send(200, {"ok": True})
        if op == "sam":
            try:
                m = sam_mask(fid, body.get("points", []), body.get("box"))
            except Exception as e:  # surface SAM errors in the UI
                return self.send(500, {"error": repr(e)})
            return self.send(200, {"png": base64.b64encode(png_bytes(m * 255)).decode()})
        self.send(404, {"error": "unknown op"})


print(f"serving {len(INDEX)} frames from {W} on http://{args.host}:{args.port}", flush=True)
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
