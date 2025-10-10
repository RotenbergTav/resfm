import os
import re
import time
import torch
import argparse
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch.nn.functional as F

HF_CKPT = "magic-leap-community/superpoint"

# ---------- conf parsing ----------
def _strip_comments(text: str) -> str:
    import re
    return re.sub(r"#.*$", "", text, flags=re.MULTILINE)

def parse_conf_scenes(conf_path: str):
    with open(conf_path, "r", encoding="utf-8") as f:
        txt = _strip_comments(f.read())
    names = []
    pat = re.compile(r"\b(train_set|validation_set|test_set)\s*=\s*\[(.*?)\]", re.DOTALL)
    for _, arr_txt in pat.findall(txt):
        names += re.findall(r'"([^"]+)"', arr_txt)
        names += re.findall(r"'([^']+)'", arr_txt)
    # dedupe keep order
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq

# ---------- HF SuperPoint (single image) ----------
_HF_PROC = None
_HF_MODEL = None

def ensure_hf(device: str):
    global _HF_PROC, _HF_MODEL
    if _HF_PROC is None:
        _HF_PROC = AutoImageProcessor.from_pretrained(HF_CKPT)
    if _HF_MODEL is None:
        _HF_MODEL = SuperPointForKeypointDetection.from_pretrained(HF_CKPT).to(device).eval()
    return _HF_PROC, _HF_MODEL

@torch.no_grad()
def extract_sparse_one(img_path: str, device: str):
    """
    HF SuperPoint (works with older transformers that don't have
    processor.post_process_keypoint_detection).
    Returns: (kpts_xy [N,2], desc_DN [256,N], scores [N], H, W)
    """
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    proc, model = ensure_hf(device)

    # Try to avoid resize/pad so mapping back to original is trivial
    try:
        inputs = proc(
            im,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
            do_pad=False,
            do_rescale=True,
            do_normalize=True,
        )
    except TypeError:
        # Older processors may not accept these kwargs
        inputs = proc(im, return_tensors="pt")

    # Move tensors to device
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    # Run model
    outputs = model(**inputs)

    # Pull processed (network) resolution
    pv = inputs.get("pixel_values")
    Hp, Wp = (pv.shape[-2], pv.shape[-1]) if isinstance(pv, torch.Tensor) else (H, W)

    # Robustly read outputs
    kp = getattr(outputs, "keypoints", None)      # [1,N,2] or [N,2] (relative or absolute)
    desc = getattr(outputs, "descriptors", None)  # [1,N,256] or [N,256]
    scr = getattr(outputs, "scores", None)        # [1,N] or [N]

    if kp is None or desc is None:
        # No detections → return empty tensors
        return (torch.zeros((0, 2), dtype=torch.float32),
                torch.zeros((256, 0), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                H, W)

    # Normalize shapes
    if kp.ndim == 3:   # [1,N,2]
        kp = kp[0]
    if desc.ndim == 3: # [1,N,256]
        desc = desc[0]
    if scr is not None and scr.ndim == 2:
        scr = scr[0]

    # Map keypoints to absolute pixel coords
    # If values look like [0,1], treat as relative to (Wp, Hp); else assume absolute on processed grid.
    if kp.numel() and float(kp.max()) <= 1.5 and float(kp.min()) >= -0.1:
        kx = kp[:, 0] * Wp
        ky = kp[:, 1] * Hp
    else:
        kx = kp[:, 0]
        ky = kp[:, 1]
    # If processor resized, rescale to original (W,H)
    scale_x = W / max(Wp, 1)
    scale_y = H / max(Hp, 1)
    kpts_xy = torch.stack([kx * scale_x, ky * scale_y], dim=1).to(torch.float32)  # [N,2]

    # Descriptors to [256,N] and L2-normalize columns
    desc_DN = desc.to(torch.float32).t().contiguous()  # [256,N]
    desc_DN = F.normalize(desc_DN, dim=0)

    scores = scr.to(torch.float32) if scr is not None else torch.zeros((0,), dtype=torch.float32)

    return kpts_xy, desc_DN, scores, H, W


# ---------- per-scene precompute (no batching) ----------
def list_images(images_dir: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(exts)]

def precompute_scene(images_dir: str, out_dir: str, device: str = "cuda", overwrite: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    ims = list_images(images_dir)
    if not ims:
        print(f"[warn] no images in {images_dir}")
        return

    t0, done = time.time(), 0
    for idx, name in enumerate(ims, 1):
        src = os.path.join(images_dir, name)
        dst = os.path.join(out_dir, name + ".spdesc.pt")

        if (not overwrite) and os.path.isfile(dst):
            if idx % 50 == 0:
                print(f"[{idx}/{len(ims)}] skip (exists): {dst}")
            continue

        try:
            kpts_xy, desc_DN, scores, H, W = extract_sparse_one(src, device)
        except Exception as e:
            print(f"[err] HF SP failed on {src}: {e}")
            continue

        payload = {
            "type": "sparse",
            "kpts":  kpts_xy.half().cpu(),    # [N,2]
            "desc":  desc_DN.half().cpu(),    # [256,N]
            "scores": scores.half().cpu(),    # [N] (may be empty)
            "H": int(H), "W": int(W),
        }
        torch.save(payload, dst)
        done += 1

        if idx % 20 == 0 or idx == len(ims):
            dt = time.time() - t0
            ips = done / max(dt, 1e-6)
            print(f"[{idx}/{len(ims)}] wrote {dst}  (scene imgs saved={done}; {ips:.2f} img/s)")

    print(f"[scene] wrote {done} files to {out_dir}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf",        required=True, help="Path to RESFM_Learning.conf")
    ap.add_argument("--images_root", required=True, help="Root with '<scene> new images' folders")
    ap.add_argument("--out_root",    required=True, help="Where to save descriptors (mirrors scene folders)")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes = parse_conf_scenes(args.conf)
    if not scenes:
        print(f"[warn] no scenes found in conf: {args.conf}")
        return

    print(f"[info] {len(scenes)} scene(s) from conf:")
    for s in scenes: print("  -", s)

    for scene in scenes:
        in_dir  = os.path.join(args.images_root, f"{scene} images")
        out_dir = os.path.join(args.out_root,    f"{scene} images")
        if not os.path.isdir(in_dir):
            print(f"[skip] images dir missing: {in_dir}")
            continue
        print(f"[precompute] {scene}")
        precompute_scene(in_dir, out_dir, device=args.device, overwrite=args.overwrite)

if __name__ == "__main__":
    main()