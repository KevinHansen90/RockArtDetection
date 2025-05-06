#!/usr/bin/env python3
"""
cluster_motifs.py  –  2025-05-05 upgrade
────────────────────────────────────────────────────────────────────────
Adds:
  • --pca <d>    : L2-normalise + PCA(d) before clustering
  • --metric     : euclidean | cosine for DBSCAN
  • DBSCAN defaults tuned to eps=0.55  min_samples=4
  • Lean DBSCAN grid {0.35,0.45,0.55} × {4,5}

Back-compatible: if you skip --pca you get the original behaviour.
"""
# ────────────────────────── stdlib & typing ──────────────────────────
import os, sys, argparse, shutil, json, itertools
from typing import Optional, List, Dict

# ───────────────────────────── 3rd-party ─────────────────────────────
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# ─────────────────────────────────────────────────────────────────────
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                             SpectralClustering, DBSCAN)
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score,
                             homogeneity_score, completeness_score)

# ───────────────────────── paths & helper ────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # …/src/clustering
_SRC_DIR      = os.path.dirname(_SCRIPT_DIR)                        # …/src
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                           # …/RockArtDetection
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from src.training.utils import get_device                       # type: ignore
except Exception:
    def get_device(dev_arg: Optional[str] = None):
        if dev_arg:
            return torch.device(dev_arg)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────── experiment grid parameters ───────────────────
FEATURE_MODELS          = ["resnet18", "resnet50", "densenet121", "vgg16"]
ALGORITHMS_WITH_K       = ["kmeans", "agglomerative", "spectral"]

# tuned DBSCAN grid (3 ε × 2 min_s  instead of 4 × 4)
DBSCAN_EPS_GRID         = [0.25, 0.30, 0.35, 0.40, 0.45]
DBSCAN_MIN_SAMPLES_GRID = [4, 5]

K_RANGE                 = range(2, 11)      # 2 … 10 clusters
BATCH_DEFAULT_ROOT      = "outputs"

# ═════════════════════ feature extractor ═════════════════════════════
def load_feature_extractor(model_name: str, device: torch.device):
    """Return (cnn without classifier head, preprocessing transform)."""
    weights = "DEFAULT"
    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    name = model_name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=weights);  model.fc = nn.Identity()
    elif name == "resnet50":
        model = models.resnet50(weights=weights);  model.fc = nn.Identity()
    elif name == "densenet121":
        model = models.densenet121(weights=weights);  model.classifier = nn.Identity()
    elif name == "vgg16":
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    elif name == "inceptionv3":
        model = models.inception_v3(weights=weights, aux_logits=True);  model.fc = nn.Identity()
        preprocess = T.Compose([
            T.Resize(299), T.CenterCrop(299), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported feature extractor: {model_name}")

    model.eval().to(device)
    print(f"[INFO] Loaded {model_name} on {device}")
    return model, preprocess

# ═════════════════════════ dataset ═══════════════════════════════════
class MotifDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir, self.transform = image_dir, transform
        self.image_files = sorted(f for f in os.listdir(image_dir)
                                  if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not self.image_files:
            raise FileNotFoundError(f"No images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        path = os.path.join(self.image_dir, name)
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, name
        except UnidentifiedImageError:
            print(f"[WARN] Corrupted image skipped: {path}", file=sys.stderr)
            return None

def collate_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, names = zip(*batch)
    return torch.stack(imgs, 0), list(names)

# ═══════════════════ extraction & clustering ═════════════════════════
def extract_features(model, loader, device):
    feats, names = [], []
    with torch.no_grad():
        for x, n in tqdm(loader, desc="feature extraction"):
            x = x.to(device, non_blocking=True)
            feats.append(model(x).cpu().numpy())
            names.extend(n)
    return np.concatenate(feats, 0), names

def preprocess_feats(feats: np.ndarray, pca_dim: int) -> np.ndarray:
    """L2-normalise + optional PCA."""
    feats = normalize(feats)
    if pca_dim and pca_dim < feats.shape[1]:
        feats = PCA(pca_dim, random_state=42).fit_transform(feats)
    return feats

def perform_clustering(features: np.ndarray,
                       algo: str, k: int,
                       eps: float = .55, min_s: int = 4,
                       metric: str = "euclidean"):
    algo = algo.lower()
    if algo == "kmeans":
        try:
            model = KMeans(k, random_state=42, n_init="auto")
        except TypeError:
            model = KMeans(k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        inertia = float(model.inertia_)
    elif algo == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(features)
        inertia = _withinss(features, labels)
    elif algo == "spectral":
        if k >= len(features):
            k = max(1, len(features) - 1)
        model = SpectralClustering(n_clusters=k, random_state=42,
                                   assign_labels="kmeans",
                                   affinity="nearest_neighbors")
        labels = model.fit_predict(features)
        inertia = _withinss(features, labels)
    elif algo == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_s, metric=metric)
        labels = model.fit_predict(features)
        inertia = None
    else:
        raise ValueError(f"Unsupported clustering algo: {algo}")
    return labels, inertia

def _withinss(X: np.ndarray, labels: np.ndarray) -> float:
    ss = 0.0
    for lab in np.unique(labels):
        pts = X[labels == lab]
        if len(pts):
            centre = pts.mean(axis=0, keepdims=True)
            ss += ((pts - centre) ** 2).sum()
    return float(ss)

# ═════════════════════── visual helpers ──────────────────────────────
def tsne_plot(feats: np.ndarray, labels: np.ndarray, out_path: str):
    tsne = TSNE(2, random_state=42, perplexity=min(30, len(feats) - 1))
    emb = tsne.fit_transform(feats)
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="viridis", s=8, alpha=.7)
    plt.title("t-SNE embedding")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

def silhouette_hbar(feats: np.ndarray, labels: np.ndarray, out_path: str):
    if len(np.unique(labels)) < 2:
        return
    svals = silhouette_samples(feats, labels)
    y_lower = 10
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        vals = np.sort(svals[labels == lab])
        size = len(vals)
        plt.barh(range(y_lower, y_lower + size), vals, height=1)
        y_lower += size + 10
    plt.axvline(np.mean(svals), color="red", ls="--")
    plt.title("Silhouette per cluster")
    plt.xlabel("silhouette coefficient"); plt.ylabel("cluster id")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()

def elbow_curve(ks: List[int], inertias: List[float], out_png: str, title: str):
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xticks(ks); plt.grid(ls=":")
    plt.title(title); plt.xlabel("k"); plt.ylabel("inertia / within-SS")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def dbscan_bar(eps_grid: List[float], silhouettes: List[float], out_png: str):
    clean = ["NA" if s == -1 else f"{s:.3f}" for s in silhouettes]
    plt.figure(figsize=(6, 4))
    plt.bar([str(e) for e in eps_grid],
            [0 if s == -1 else s for s in silhouettes],
            color=["lightgray" if s == -1 else "steelblue" for s in silhouettes])
    plt.title("DBSCAN – best silhouette per ε (gray = noise)")
    plt.xlabel("eps"); plt.ylabel("silhouette")
    for i, txt in enumerate(clean):
        plt.text(i, 0.02, txt, ha="center", va="bottom", fontsize=8, rotation=90)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ═════════════════════ core single experiment ════════════════════════
def run_single(input_dir: str, output_dir: str,
               model_name: str, algo: str,
               k: int = 2,
               eps: float = .55, min_samples: int = 4,
               metric: str = "euclidean", pca_dim: int = 0,
               batch_size: int = 32, copy_images: bool = False,
               device_override: Optional[str] = None,
               make_plots: bool = True,
               true_label_csv: Optional[str] = None) -> Dict[str, float]:

    device = torch.device(device_override) if device_override else get_device()
    model, tfm = load_feature_extractor(model_name, device)

    dl = DataLoader(MotifDataset(input_dir, tfm),
                    batch_size=batch_size, shuffle=False,
                    collate_fn=collate_filter_none,
                    pin_memory=(device.type == "cuda"))

    feats, names = extract_features(model, dl, device)
    feats = preprocess_feats(feats, pca_dim)
    labels, inertia = perform_clustering(feats, algo, k,
                                         eps=eps, min_s=min_samples,
                                         metric=metric)

    os.makedirs(output_dir, exist_ok=True)

    # ───────────── metrics ─────────────
    metrics: Dict[str, float] = {}
    core = labels != -1
    metrics["noise_ratio"] = float((~core).sum()) / len(labels)

    unique_core = np.unique(labels[core])
    if len(unique_core) >= 2:
        metrics["silhouette"]        = float(silhouette_score(feats[core], labels[core]))
        metrics["davies_bouldin"]    = float(davies_bouldin_score(feats[core], labels[core]))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(feats[core], labels[core]))
    else:
        metrics.update(dict(silhouette=-1,
                            davies_bouldin=float("inf"),
                            calinski_harabasz=-1))

    if inertia is not None:
        metrics["inertia"] = inertia

    if true_label_csv and os.path.exists(true_label_csv):
        true      = pd.read_csv(true_label_csv)["label"].values
        core_true = true[core]; core_pred = labels[core]
        metrics["homogeneity"]  = float(homogeneity_score(core_true, core_pred))
        metrics["completeness"] = float(completeness_score(core_true, core_pred))

    json.dump(metrics, open(os.path.join(output_dir, "metrics.json"), "w"), indent=2)

    pd.DataFrame({"filename": names, "cluster": labels}) \
      .to_csv(os.path.join(output_dir, "assignments.csv"), index=False)

    # ─────────── optional artefacts ───────────
    if make_plots:
        tsne_plot(feats, labels, os.path.join(output_dir, "tsne.png"))
        silhouette_hbar(feats, labels, os.path.join(output_dir, "silhouette.png"))

    if copy_images:
        for lab in np.unique(labels):
            os.makedirs(os.path.join(output_dir, f"cluster_{lab}"), exist_ok=True)
        for f, lab in zip(names, labels):
            shutil.copy2(os.path.join(input_dir, f),
                         os.path.join(output_dir, f"cluster_{lab}", f))

    return metrics

# ═════════════════════ batch grid driver ═════════════════════════════
def run_full_grid(input_dir: str, root_out: str,
                  batch_size: int = 32, device_override: Optional[str] = None,
                  pca_dim: int = 0, metric: str = "euclidean"):

    os.makedirs(root_out, exist_ok=True)
    global_rows = []

    for model_name in FEATURE_MODELS:
        for algo in itertools.chain(ALGORITHMS_WITH_K, ["dbscan"]):
            algo_root = os.path.join(root_out, model_name, algo)
            os.makedirs(algo_root, exist_ok=True)

            if algo in ALGORITHMS_WITH_K:
                inertias = []
                for k in K_RANGE:
                    out_dir = os.path.join(algo_root, f"k_{k}")
                    m = run_single(input_dir, out_dir, model_name, algo,
                                   k=k, batch_size=batch_size,
                                   pca_dim=pca_dim, metric=metric,
                                   device_override=device_override,
                                   make_plots=True, copy_images=False)
                    m.update(dict(model=model_name, algo=algo, k=k))
                    global_rows.append(m); inertias.append(m.get("inertia", np.nan))
                elbow_curve(list(K_RANGE), inertias,
                            os.path.join(algo_root, "elbow_curve.png"),
                            f"{model_name} – {algo}")
                pd.DataFrame(global_rows).query("model==@model_name and algo==@algo") \
                  .to_csv(os.path.join(algo_root, "summary_metrics.csv"), index=False)

            else:  # ── tuned DBSCAN grid ──
                best_for_eps: Dict[float, float] = {e: -1 for e in DBSCAN_EPS_GRID}
                for eps, min_s in itertools.product(DBSCAN_EPS_GRID,
                                                    DBSCAN_MIN_SAMPLES_GRID):
                    out_dir = os.path.join(algo_root, f"eps_{eps}_ms_{min_s}")
                    m = run_single(input_dir, out_dir, model_name, algo,
                                   eps=eps, min_samples=min_s, k=1,
                                   batch_size=batch_size,
                                   pca_dim=pca_dim, metric=metric,
                                   device_override=device_override,
                                   make_plots=True, copy_images=False)
                    m.update(dict(model=model_name, algo=algo,
                                  eps=eps, min_samples=min_s))
                    global_rows.append(m)
                    if m["silhouette"] > best_for_eps[eps]:
                        best_for_eps[eps] = m["silhouette"]
                eps_vals = sorted(best_for_eps.keys())
                sil_vals = [best_for_eps[e] for e in eps_vals]
                dbscan_bar(eps_vals, sil_vals,
                           os.path.join(algo_root, "eps_silhouette.png"))
                pd.DataFrame(global_rows).query("model==@model_name and algo==@algo") \
                  .to_csv(os.path.join(algo_root, "summary_metrics.csv"), index=False)

    pd.DataFrame(global_rows).to_csv(os.path.join(root_out, "global_metrics.csv"), index=False)
    print(f"[DONE] All experiments finish – results under {root_out}")

# ═════════════════════── CLI parsing & main ──────────────────────────
def build_cli():
    p = argparse.ArgumentParser("cluster_motifs")
    p.add_argument("--input", required=True, help="Directory with cropped motif images")

    # batch / single mode
    p.add_argument("--root", default=BATCH_DEFAULT_ROOT,
                   help=f"Root output folder (default: {BATCH_DEFAULT_ROOT})")
    p.add_argument("--no-batch", action="store_true",
                   help="Run a *single* experiment instead of the full grid")

    # single-run only
    p.add_argument("--output")
    p.add_argument("--feature-model", choices=FEATURE_MODELS)
    p.add_argument("--cluster-algo", choices=ALGORITHMS_WITH_K + ["dbscan"])
    p.add_argument("--num-clusters", type=int, help="k for algorithms that need it")
    p.add_argument("--dbscan_eps", type=float, default=0.55)
    p.add_argument("--dbscan_min_samples", type=int, default=4)
    p.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--pca", type=int, default=0,
                   help="Apply PCA to this dimensionality (0 = off)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", help="cuda | cpu | mps")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy cropped motifs into cluster_* folders (single run)")
    return p

def main():
    args = build_cli().parse_args()

    if args.no_batch:
        # ── single experiment mode ──
        required = [args.output, args.feature_model, args.cluster_algo]
        if args.cluster_algo in ALGORITHMS_WITH_K:
            required.append(args.num_clusters)
        if any(v is None for v in required):
            print("[ERROR] Missing arguments for single-run.", file=sys.stderr); sys.exit(1)

        run_single(input_dir=args.input, output_dir=args.output,
                   model_name=args.feature_model, algo=args.cluster_algo,
                   k=args.num_clusters or 2,
                   eps=args.dbscan_eps, min_samples=args.dbscan_min_samples,
                   metric=args.metric, pca_dim=args.pca,
                   batch_size=args.batch_size, copy_images=args.copy_images,
                   device_override=args.device, make_plots=True)
    else:
        # ── full batch grid ──
        run_full_grid(input_dir=args.input, root_out=args.root,
                      batch_size=args.batch_size, device_override=args.device,
                      pca_dim=args.pca, metric=args.metric)

if __name__ == "__main__":
    main()

