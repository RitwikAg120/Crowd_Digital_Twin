"""
Train the dense-crowd head-point model (dense.P2PNet) — run this on the GB10.

    python train_dense.py --data crowdhuman:dataset/CrowdHuman \\
                          --init imagenet --epochs 100 --amp --out weights/p2pnet_crowd.pth

    # then fine-tune on your own annotated frames (points format)
    python train_dense.py --data points:dataset/kumbh_points --init weights/p2pnet_crowd.pth \\
                          --epochs 60 --lr 5e-5 --out weights/p2pnet_crowd_kumbh.pth

Datasets (--data kind:path, repeatable; train on their train split, validate
on val/test):
  crowdhuman  CrowdHuman folder: annotation_train.odgt, annotation_val.odgt and the
              images (anywhere below it, e.g. Images/). Every person's head box
              (hbox) centre is a head point; "mask" regions (unlabelled crowds) and
              heads marked ignore are blanked out, so the model isn't taught that
              people there are background.
  points      your own: {train,test}/<name>.jpg + <name>.txt with one "x y" head point
              per line (tools/annotate_points.py writes these)

Init: "imagenet" (VGG16-BN ImageNet weights, downloaded by torchvision),
"none" (random, for smoke tests), or a checkpoint path (a previous run). The
loss follows P2PNet: one-to-one Hungarian matching of predicted points to
heads (cost = 0.05·distance − head score), cross-entropy with background
weight 0.5, and point MSE × 2e-4. The best checkpoint by validation MAE is
written to --out, ready for Config.DENSE_MODEL.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from dense import P2PNet, load_p2pnet

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)
MEAN_BGR = tuple(int(round(255 * m)) for m in MEAN[::-1])     # what masked regions are filled with
IMG_EXT = (".jpg", ".jpeg", ".png")


# ─── Datasets ─────────────────────────────────────────────────────────────────

def _xywh_to_xyxy(b):
    x, y, w, h = (float(v) for v in b[:4])
    return [x, y, x + w, y + h]


def crowdhuman_samples(root: Path, split: str):
    """
    CrowdHuman: [(image, (N, 2) head centres, [ignore boxes x1 y1 x2 y2])].
    The .odgt files hold one JSON record per image; images are found by ID
    anywhere below `root` (the zips unpack to Images/ by default).
    """
    name = "annotation_train.odgt" if split == "train" else "annotation_val.odgt"
    odgt = next(iter(sorted(root.rglob(name))), None)
    if odgt is None:
        return []
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT}
    out = []
    with open(odgt, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            img = images.get(rec["ID"])
            if img is None:
                continue
            heads, ignore = [], []
            for b in rec.get("gtboxes", []):
                if b.get("tag") == "mask":                              # crowd region, not labelled
                    ignore.append(_xywh_to_xyxy(b.get("fbox") or b.get("vbox")))
                    continue
                if b.get("tag") != "person" or "hbox" not in b:
                    continue
                hb = _xywh_to_xyxy(b["hbox"])
                if b.get("head_attr", {}).get("ignore", 0):             # head not clearly visible
                    ignore.append(hb)
                    continue
                heads.append([(hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2])
            out.append((img, np.asarray(heads, float).reshape(-1, 2), ignore))
    return out


def list_samples(kind: str, root: str, split: str):
    """
    [(image path, (N, 2) head points, [ignore boxes])] for one dataset split
    ("train" or "test").
    """
    r = Path(root)
    if kind == "crowdhuman":
        return crowdhuman_samples(r, split)
    if kind == "points":
        d = r / ("train" if split == "train" else "test")
        out = []
        for img in sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT):
            txt = img.with_suffix(".txt")
            pts = np.loadtxt(txt, ndmin=2)[:, :2] if txt.exists() and txt.stat().st_size else np.zeros((0, 2))
            out.append((img, pts, []))
        return out
    raise SystemExit(f"Unknown dataset kind {kind!r} (use crowdhuman or points)")


def load_image(path, pts, max_side: int, ignore=()):
    """The image with ignore boxes blanked (mean colour), scaled to ≤ max_side, and its points."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read {path}")
    for x1, y1, x2, y2 in ignore:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), MEAN_BGR, -1)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        pts = pts * s
    inside = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < img.shape[1]) & (pts[:, 1] < img.shape[0])
    return img, pts[inside]


class CrowdPatches(torch.utils.data.Dataset):
    """Random-scaled, flipped crops (PATCHES per image) with the heads inside each crop."""
    def __init__(self, samples, crop: int, patches: int, max_side: int):
        self.samples, self.crop, self.patches, self.max_side = samples, crop, patches, max_side

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, pts, ignore = self.samples[i]
        img, pts = load_image(path, pts, self.max_side, ignore)
        s = random.uniform(0.7, 1.3)
        h, w = img.shape[:2]
        if min(h, w) * s < self.crop:
            s = self.crop / min(h, w)
        img = cv2.resize(img, (max(self.crop, int(w * s)), max(self.crop, int(h * s))))
        sy, sx = img.shape[0] / h, img.shape[1] / w
        pts = pts * [sx, sy]
        x = (img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        H, W = x.shape[:2]
        crops, targets = [], []
        for _ in range(self.patches):
            y0, x0 = random.randint(0, H - self.crop), random.randint(0, W - self.crop)
            c = x[y0:y0 + self.crop, x0:x0 + self.crop]
            p = pts - [x0, y0]
            p = p[(p[:, 0] >= 0) & (p[:, 0] < self.crop) & (p[:, 1] >= 0) & (p[:, 1] < self.crop)]
            if random.random() < 0.5:
                c = c[:, ::-1]
                p = p.copy()
                p[:, 0] = self.crop - 1 - p[:, 0]
            crops.append(torch.from_numpy(np.ascontiguousarray(c.transpose(2, 0, 1))))
            targets.append(torch.as_tensor(p, dtype=torch.float32).reshape(-1, 2))
        return torch.stack(crops), targets


def collate(batch):
    return torch.cat([b[0] for b in batch]), [t for b in batch for t in b[1]]


# ─── Loss ─────────────────────────────────────────────────────────────────────

def p2p_loss(logits, points, targets, cost_point=0.05, eos_coef=0.5, point_coef=2e-4):
    """P2PNet's one-to-one matching loss for a batch of crops."""
    prob = logits.float().softmax(-1)
    tgt_cls = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
    src_pts, tgt_pts = [], []
    with torch.no_grad():
        for b, t in enumerate(targets):
            if len(t) == 0:
                continue
            t = t.to(logits.device)
            C = cost_point * torch.cdist(points[b].float(), t) - prob[b, :, 1:2]
            qi, ti = linear_sum_assignment(C.cpu().numpy())
            tgt_cls[b, qi] = 1
            src_pts.append((b, torch.as_tensor(qi, device=logits.device),
                            t[torch.as_tensor(ti, device=logits.device)]))
    weight = torch.tensor([eos_coef, 1.0], device=logits.device)
    loss_ce = F.cross_entropy(logits.float().transpose(1, 2), tgt_cls, weight)
    n = max(sum(len(t) for t in targets), 1)
    if src_pts:
        pred = torch.cat([points[b, qi].float() for b, qi, _ in src_pts])
        tgt  = torch.cat([t for _, _, t in src_pts])
        loss_pt = F.mse_loss(pred, tgt, reduction="sum") / n
    else:
        loss_pt = points.sum() * 0.0
    return loss_ce + point_coef * loss_pt, loss_ce.item(), loss_pt.item()


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, samples, device, max_side: int, threshold: float = 0.5, amp=False):
    model.eval()
    err = []
    for path, pts, ignore in samples:
        img, pts = load_image(path, pts, max_side, ignore)
        h, w = img.shape[:2]
        x = (img[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
        x = np.pad(x, ((0, (128 - h % 128) % 128), (0, (128 - w % 128) % 128), (0, 0)))
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            logits, p = model(t)
        score = logits.float().softmax(-1)[0, :, 1]
        inside = (p[0, :, 0] < w) & (p[0, :, 1] < h)
        err.append(float(((score >= threshold) & inside).sum().item()) - len(pts))
    model.train()
    err = np.array(err)
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_model(init: str, device: str) -> P2PNet:
    if init == "imagenet":
        import torchvision
        feats = torchvision.models.vgg16_bn(weights="IMAGENET1K_V1").features
        return P2PNet(backbone_features=feats).to(device)
    if init == "none":
        return P2PNet().to(device)
    return load_p2pnet(init, device).train()


def main():
    ap = argparse.ArgumentParser(description="Train the dense-crowd head-point model (P2PNet)")
    ap.add_argument("--data", action="append", required=True, help="kind:path (repeatable)")
    ap.add_argument("--init", default="imagenet", help="imagenet | none | checkpoint path")
    ap.add_argument("--out", default="weights/p2pnet_crowd.pth")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="images per step (× --patches crops)")
    ap.add_argument("--patches", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--max-side", type=int, default=2048, help="downscale larger images")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N validations without a better MAE (0 = never); "
                         "use with --val-every 1 to mean N epochs")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.last.pth (saved after every epoch)")
    ap.add_argument("--val-max", type=int, default=0, help="validate on at most N images (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (CUDA)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val = [], []
    for spec in args.data:
        kind, root = spec.split(":", 1)
        tr, te = list_samples(kind, root, "train"), list_samples(kind, root, "test")
        print(f"  {kind}: {len(tr)} train / {len(te)} validation images from {root}")
        train += tr
        val += te
    if not train:
        raise SystemExit("No training images found.")
    if args.val_max and len(val) > args.val_max:
        val = random.sample(val, args.val_max)

    model = build_model(args.init, args.device)
    backbone = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    rest     = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                             {"params": backbone, "lr": args.lr_backbone}], weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        CrowdPatches(train, args.crop, args.patches, args.max_side), batch_size=args.batch,
        shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=len(train) > args.batch,
        persistent_workers=args.workers > 0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    amp = args.amp and args.device.startswith("cuda")
    best, step, log, stale, start = math.inf, 0, [], 0, 1
    last_path = out.with_suffix(".last.pth")
    if args.resume and last_path.exists():
        st = torch.load(str(last_path), map_location=args.device, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        best, stale, start = st["best"], st["stale"], st["epoch"] + 1
        log = json.loads(out.with_suffix(".log.json").read_text()) if out.with_suffix(".log.json").exists() else []
        print(f"  resuming after epoch {st['epoch']} (best MAE {best:.1f}, {stale} without improvement)")
    t0 = time.time()
    epoch = start - 1
    for epoch in range(start, args.epochs + 1):
        model.train()
        losses = []
        for x, targets in loader:
            x = x.to(args.device, non_blocking=True)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
                logits, pts = model(x)
            loss, lce, lpt = p2p_loss(logits, pts, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            losses.append((loss.item(), lce, lpt))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        l = np.mean(losses, axis=0) if losses else [float("nan")] * 3
        msg = f"epoch {epoch}: loss {l[0]:.4f} (ce {l[1]:.4f}, point {l[2]:.1f}) {time.time() - t0:.0f}s"
        last = args.max_steps and step >= args.max_steps
        if val and (epoch % args.val_every == 0 or epoch == args.epochs or last):
            mae, rmse = validate(model, val, args.device, args.max_side, amp=amp)
            msg += f" | val MAE {mae:.1f} RMSE {rmse:.1f}"
            log.append({"epoch": epoch, "mae": mae, "rmse": rmse, "loss": float(l[0])})
            if mae < best:
                best, stale = mae, 0
                torch.save({"model": model.state_dict(), "row": 2, "epoch": epoch, "mae": mae,
                            "rmse": rmse, "data": args.data}, out)
                msg += "  → saved"
            else:
                stale += 1
                msg += f"  (no improvement: {stale}/{args.patience or '∞'})"
            out.with_suffix(".log.json").write_text(json.dumps(log, indent=2))
        print(msg, flush=True)
        # Everything needed to resume after this epoch
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best": best, "stale": stale}, last_path)
        if args.patience and stale >= args.patience:
            print(f"Stopping early: validation MAE hasn't improved for {stale} checks.", flush=True)
            break
        if last:
            break
    if not val:
        torch.save({"model": model.state_dict(), "row": 2, "epoch": epoch, "data": args.data}, out)
    (out.with_suffix(".log.json")).write_text(json.dumps(log, indent=2))
    print(f"Best validation MAE {best:.1f} → {out}")


if __name__ == "__main__":
    main()
