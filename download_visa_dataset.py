"""
download_visa.py

Télécharge le dataset VisA (Visual Anomaly), l'extrait, et le réorganise
au même format que MVTec AD :

    data/visa/
        <category>/
            train/good/*.png
            test/
                good/*.png
                bad/*.png
            ground_truth/
                bad/*_mask.png

Ce format est directement compatible avec la classe MVTecAD existante
(defect_type sera "good" ou "bad" -- VisA ne fournit pas de sous-types
de défauts détaillés comme MVTec AD, contrairement à ce dernier).

Usage:
    python download_visa.py --output data/visa
"""

import argparse
import csv
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPLIT_CSV_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"

CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def download_file(url, dest_path, min_size_mb=1):
    """Télécharge un fichier avec une barre de progression simple, et vérifie sa taille."""
    if dest_path.exists():
        print(f"Déjà présent : {dest_path}")
        return

    print(f"Téléchargement de {url} ...")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  {percent:.1f}%", end="")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
    print()

    size_mb = dest_path.stat().st_size / (1024 ** 2)
    print(f"  Taille téléchargée : {size_mb:.1f} Mo")
    if size_mb < min_size_mb:
        raise ValueError(
            f"Fichier suspect (trop petit, {size_mb:.2f} Mo) : "
            f"le lien est peut-être expiré ou invalide."
        )


def extract_tar(tar_path, dest_dir):
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"Déjà extrait : {dest_dir}")
        return
    print(f"Extraction de {tar_path} vers {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir)


def load_split(split_csv_path):
    """
    Charge le split officiel 1-class (train/test).
    Colonnes du CSV : object, split, label, image, mask
    (split = "train"/"test", label = "normal"/"anomaly")
    """
    split = {}  # category -> {"train": [...], "test": [...]}
    with open(split_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["object"]
            split.setdefault(category, {"train": [], "test": []})
            entry = {
                "label": row["label"],           # "normal" ou "anomaly"
                "image_path": row["image"],
                "mask_path": row.get("mask", "") or None,
            }
            split[category][row["split"]].append(entry)
    return split


def reorganize_category(category, raw_root, split, output_root):
    """
    Réorganise une catégorie du format VisA brut vers le format MVTec-AD-like.
    """
    cat_split = split[category]

    train_good_dir = output_root / category / "train" / "good"
    test_good_dir = output_root / category / "test" / "good"
    test_bad_dir = output_root / category / "test" / "bad"
    gt_bad_dir = output_root / category / "ground_truth" / "bad"

    for d in [train_good_dir, test_good_dir, test_bad_dir, gt_bad_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Train (uniquement des images normales) ---
    for i, entry in enumerate(cat_split["train"]):
        src = raw_root / entry["image_path"]
        dst = train_good_dir / f"{i:04d}.png"
        _copy_as_png(src, dst)

    # --- Test (normal + anomalies) ---
    idx_good, idx_bad = 0, 0
    for entry in cat_split["test"]:
        src = raw_root / entry["image_path"]

        if entry["label"] == "normal":
            dst = test_good_dir / f"{idx_good:04d}.png"
            _copy_as_png(src, dst)
            idx_good += 1
        else:
            dst = test_bad_dir / f"{idx_bad:04d}.png"
            _copy_as_png(src, dst)

            if entry["mask_path"]:
                mask_src = raw_root / entry["mask_path"]
                mask_dst = gt_bad_dir / f"{idx_bad:04d}_mask.png"
                _copy_mask_as_binary_png(mask_src, mask_dst)

            idx_bad += 1

    print(f"  {category} : train={len(cat_split['train'])}, "
          f"test_good={idx_good}, test_bad={idx_bad}")


def _copy_as_png(src, dst):
    """Copie une image en la convertissant en PNG (les images VisA sont en .JPG)."""
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    img.save(dst)


def _copy_mask_as_binary_png(src, dst):
    """Copie un masque en le binarisant (0 = normal, 255 = anomalie)."""
    if dst.exists():
        return
    mask = Image.open(src).convert("L")
    mask_np = np.array(mask)
    mask_bin = np.where(mask_np > 0, 255, 0).astype(np.uint8)
    Image.fromarray(mask_bin).save(dst)


def main():
    parser = argparse.ArgumentParser(description="Télécharge et réorganise le dataset VisA")
    parser.add_argument("--output", type=str, default="data/visa",
                         help="Dossier de sortie final (format MVTec-AD-like)")
    parser.add_argument("--download_dir", type=str, default="data/visa_raw",
                         help="Dossier temporaire pour le téléchargement/extraction brute")
    parser.add_argument("--keep_raw", action="store_true",
                         help="Garder les fichiers bruts téléchargés après réorganisation")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    output_root = Path(args.output)

    tar_path = download_dir / "VisA_20220922.tar"
    extracted_dir = download_dir / "VisA"
    split_csv_path = download_dir / "1cls.csv"

    # 1. Téléchargement
    download_file(VISA_URL, tar_path, min_size_mb=100)
    download_file(SPLIT_CSV_URL, split_csv_path, min_size_mb=0.001)

    # 2. Extraction
    extract_tar(tar_path, extracted_dir)

    # La racine réelle des catégories peut être extracted_dir ou extracted_dir/VisA
    # selon la structure interne du tar -- on vérifie :
    raw_root = extracted_dir
    if not (raw_root / "candle").exists() and (raw_root / "VisA").exists():
        raw_root = raw_root / "VisA"

    # 3. Chargement du split officiel
    print("Chargement du split 1-class...")
    split = load_split(split_csv_path)

    # 4. Réorganisation par catégorie
    print("Réorganisation au format MVTec-AD-like...")
    for category in CATEGORIES:
        if category not in split:
            print(f"  ! Catégorie absente du split : {category}, ignorée")
            continue
        reorganize_category(category, raw_root, split, output_root)

    # 5. Nettoyage optionnel
    if not args.keep_raw:
        print("Nettoyage des fichiers bruts...")
        if tar_path.exists():
            os.remove(tar_path)

    print(f"\nTerminé. Dataset prêt dans : {output_root}")
    print("Catégories :", sorted(os.listdir(output_root)))


if __name__ == "__main__":
    main()