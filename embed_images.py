"""
extract_embeddings.py

Script pour extraire des embeddings de type PatchCore (WideResNet-50, layer2+layer3)
sur le dataset MVTec AD, et les sauvegarder sur disque pour réutilisation ultérieure.

Usage:
    python extract_embeddings.py --category bottle --root data/mv_tec_ad --output embeddings/

    # Pour toutes les catégories d'un coup :
    python extract_embeddings.py --category all --root data/mv_tec_ad --output embeddings/
"""

import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MVTecAD(Dataset):
    """
    Structure attendue :
    data/mv_tec_ad/
        <category>/
            train/
                good/*.png
            test/
                good/*.png
                <anomaly_type_1>/*.png
                <anomaly_type_2>/*.png
                ...
            ground_truth/
                <anomaly_type_1>/*_mask.png
                ...
    """

    def __init__(self, root, category, split="train", transform=None, image_size=256):
        self.root = Path(root) / category
        self.split = split
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        self.image_size = image_size
        self.samples = []  # (image_path, label, defect_type, mask_path)

        if split == "train":
            good_dir = self.root / "train" / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append((img_path, 0, "good", None))

        elif split == "test":
            test_dir = self.root / "test"
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue

                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1

                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if label == 1:
                        candidate = (
                            self.root / "ground_truth" / defect_type /
                            f"{img_path.stem}_mask.png"
                        )
                        mask_path = candidate if candidate.exists() else None

                    self.samples.append((img_path, label, defect_type, mask_path))
        else:
            raise ValueError(f"split doit être 'train' ou 'test', reçu : {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, defect_type, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = T.Resize((self.image_size, self.image_size))(mask)
            mask = T.ToTensor()(mask)
        else:
            mask = torch.zeros((1, self.image_size, self.image_size))

        return {
            "image": image,
            "label": label,
            "defect_type": defect_type,
            "mask": mask,
            "path": str(img_path),
        }


# ---------------------------------------------------------------------------
# Extraction des embeddings (PatchCore-style)
# ---------------------------------------------------------------------------

def embed_patches(features, patch_size=3):
    """Agrège l'info locale autour de chaque position (voisinage p, stride=1)."""
    pooled = []
    for f in features:
        f = F.avg_pool2d(f, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        pooled.append(f)
    return pooled


def combine_layers(pooled_features):
    """Aligne spatialement les feature maps sur la résolution la plus grande, puis concatène."""
    target_size = pooled_features[0].shape[-2:]
    resized = [pooled_features[0]]

    for f in pooled_features[1:]:
        f_resized = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
        resized.append(f_resized)

    combined = torch.cat(resized, dim=1)  # [B, C_total, H, W]
    return combined


def extract_patch_embeddings(backbone, images, patch_size=3):
    """Retourne des embeddings de forme [B, H*W, C]."""
    with torch.no_grad():
        features = backbone(images)
        pooled = embed_patches(features, patch_size)
        combined = combine_layers(pooled)

    B, C, H, W = combined.shape
    embeddings = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return embeddings, (B, H, W)


def build_backbone(model_name="wide_resnet50_2", out_indices=(2, 3), device="cpu"):
    backbone = timm.create_model(
        model_name, pretrained=True, features_only=True, out_indices=out_indices
    ).to(device).eval()
    return backbone


# ---------------------------------------------------------------------------
# Pipeline d'extraction + sauvegarde pour un split complet
# ---------------------------------------------------------------------------

def extract_and_save(dataset, backbone, output_path, device="cpu",
                      batch_size=16, patch_size=3, num_workers=4):
    """
    Parcourt tout le dataset, extrait les embeddings, et sauvegarde tout
    (embeddings + métadonnées) dans un seul fichier .pt.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_embeddings = []
    all_labels = []
    all_defect_types = []
    all_paths = []
    grid_hw = None

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        defect_types = batch["defect_type"]
        paths = batch["path"]

        embeddings, (B, H, W) = extract_patch_embeddings(backbone, images, patch_size=patch_size)
        grid_hw = (H, W)

        all_embeddings.append(embeddings.cpu())
        all_labels.extend(labels.tolist())
        all_defect_types.extend(defect_types)
        all_paths.extend(paths)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [N_images, H*W, C]

    payload = {
        "embeddings": embeddings_tensor,       # [N, H*W, C]
        "labels": torch.tensor(all_labels),    # [N]
        "defect_types": all_defect_types,      # list[str], longueur N
        "paths": all_paths,                    # list[str], longueur N
        "grid_hw": grid_hw,                    # (H, W) de la grille de patchs
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"  -> {output_path}  ({embeddings_tensor.shape[0]} images, "
          f"{embeddings_tensor.shape[1]} patches/image, {embeddings_tensor.shape[2]} dims)")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def main():
    parser = argparse.ArgumentParser(description="Extraction d'embeddings PatchCore pour MVTec AD")
    parser.add_argument("--root", type=str, default="data/mv_tec_ad",
                         help="Racine du dataset MVTec AD")
    parser.add_argument("--category", type=str, default="bottle",
                         help="Catégorie MVTec AD, ou 'all' pour toutes les traiter")
    parser.add_argument("--output", type=str, default="data/embeddings",
                         help="Dossier de sortie pour les fichiers .pt")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--out_indices", type=int, nargs="+", default=[2, 3],
                         help="Indices des couches à extraire (2,3 = layer2,layer3)")
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    categories = ALL_CATEGORIES if args.category == "all" else [args.category]

    print(f"Device : {args.device}")
    print(f"Backbone : {args.backbone}, out_indices={tuple(args.out_indices)}")

    backbone = build_backbone(
        model_name=args.backbone,
        out_indices=tuple(args.out_indices),
        device=args.device,
    )

    output_dir = Path(args.output)

    for category in categories:
        print(f"\n=== Catégorie : {category} ===")

        train_dataset = MVTecAD(root=args.root, category=category, split="train",
                                 image_size=args.image_size)
        test_dataset = MVTecAD(root=args.root, category=category, split="test",
                                image_size=args.image_size)

        print(f"Train : {len(train_dataset)} images | Test : {len(test_dataset)} images")
        defect_counts = Counter(s[2] for s in test_dataset.samples)
        print(f"Répartition test : {dict(defect_counts)}")

        print("Extraction train...")
        extract_and_save(
            train_dataset, backbone,
            output_path=output_dir / args.backbone / category / "train.pt",
            device=args.device, batch_size=args.batch_size,
            patch_size=args.patch_size, num_workers=args.num_workers,
        )

        print("Extraction test...")
        extract_and_save(
            test_dataset, backbone,
            output_path=output_dir / args.backbone / category / "test.pt",
            device=args.device, batch_size=args.batch_size,
            patch_size=args.patch_size, num_workers=args.num_workers,
        )

    print("\nTerminé.")


if __name__ == "__main__":
    main()