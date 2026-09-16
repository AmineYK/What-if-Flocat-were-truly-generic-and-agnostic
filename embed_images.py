"""
extract_embeddings.py

Script pour extraire des embeddings patch-level sur le dataset MVTec AD,
avec trois backbones/modes possibles :
    - ResNet/WideResNet (style PatchCore : layer2 + layer3)
    - ViT (tokens de patchs internes du ViT, avant la tête, sans [CLS]/registres,
      ou embedding CLS global de l'image via --is_cls)
    - ViT avec patchs définis manuellement : on découpe nous-mêmes l'image en
      patchs (sans overlap), chaque patch est passé indépendamment dans le ViT,
      et on récupère le token [CLS] de chaque patch. La séquence de ces CLS
      constitue l'embedding complet de l'image.

Les embeddings sont sauvegardés sur disque (.pt) pour réutilisation ultérieure.

Usage:
    # Backbone ResNet (par défaut)
    python extract_embeddings.py --model_type resnet --category bottle \
        --root data/mv_tec_ad --output data/embeddings

    # Backbone ViT — patch tokens natifs
    python extract_embeddings.py --model_type vit --vit_model_name vit_base_patch16_224 \
        --category bottle --root data/mv_tec_ad --output data/embeddings

    # Backbone ViT — embedding CLS global
    python extract_embeddings.py --model_type vit --is_cls \
        --category bottle --root data/mv_tec_ad --output data/embeddings

    # ViT avec patchs définis manuellement (CLS par patch)
    python extract_embeddings.py --model_type vit_manual_patches \
        --manual_patch_size 64 --vit_input_size 224 \
        --category bottle --root data/mv_tec_ad --output data/embeddings

    # Toutes les catégories
    python extract_embeddings.py --model_type vit_manual_patches --category all \
        --root data/mv_tec_ad --output data/embeddings
"""

import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
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
# Extraction ResNet / WideResNet (PatchCore-style : layer2 + layer3)
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


def extract_resnet_patch_embeddings(model, images, patch_size=3):
    """Retourne des embeddings de forme [B, H*W, C] à partir d'un backbone ResNet/WideResNet."""
    with torch.no_grad():
        features = model(images)
        pooled = embed_patches(features, patch_size)
        combined = combine_layers(pooled)

    B, C, H, W = combined.shape
    embeddings = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return embeddings, (B, H, W)


def build_resnet_backbone(model_name="wide_resnet50_2", out_indices=(2, 3), device="cpu"):
    backbone = timm.create_model(
        model_name, pretrained=True, features_only=True, out_indices=out_indices
    ).to(device).eval()
    return backbone


# ---------------------------------------------------------------------------
# Extraction ViT — tokens de patchs internes ou embedding CLS global
# ---------------------------------------------------------------------------

def build_vit_model(model_name="vit_base_patch16_224", image_size=256, device="cpu"):
    """
    Charge un ViT pré-entraîné via timm. `img_size` force la résolution
    d'entrée (timm interpole le positional embedding automatiquement si
    la résolution diffère de celle du pré-entraînement d'origine).
    """
    model = timm.create_model(
        model_name, pretrained=True, img_size=image_size
    ).to(device).eval()
    return model


def build_vit_transform(model, image_size=256):
    config = resolve_data_config({}, model=model)
    config["input_size"] = (3, image_size, image_size)
    return create_transform(**config)


def extract_vit_embeddings(model, images, is_cls=False):
    """
    Extrait les embeddings d'un ViT (timm).

    - is_cls=False : retourne les embeddings des patchs internes du ViT
                     [B, H*W, C] + (B, H, W)
    - is_cls=True  : retourne l'embedding CLS (résume l'image entière)
                     [B, C]
    """
    with torch.no_grad():
        x = model.patch_embed(images)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)

        for block in model.blocks:
            x = block(x)

        x = model.norm(x)

    num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)

    if is_cls:
        cls_embedding = x[:, 0, :]  # [B, C]
        return cls_embedding

    patch_tokens = x[:, num_prefix_tokens:, :]  # [B, N, C]

    B, N, C = patch_tokens.shape
    H = W = int(round(N ** 0.5))
    if H * W != N:
        raise ValueError(
            f"Le nombre de patchs ({N}) n'est pas un carré parfait : "
            f"grille spatiale non reconstructible proprement."
        )

    return patch_tokens, (B, H, W)


# ---------------------------------------------------------------------------
# Extraction ViT avec patchs définis manuellement (CLS par patch)
# ---------------------------------------------------------------------------

def split_into_patches(images, patch_size):
    """
    Découpe un batch d'images [B, C, H, W] en patchs non chevauchants de
    taille (patch_size x patch_size), via unfold (stride = patch_size).

    Retourne un tenseur [B, num_patches, C, patch_size, patch_size] et
    la grille spatiale (nH, nW) des patchs.
    """
    B, C, H, W = images.shape

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"L'image ({H}x{W}) n'est pas divisible par manual_patch_size={patch_size}. "
            f"Choisis un patch_size qui divise image_size, ou ajuste --image_size."
        )

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches: [B, C, nH, nW, patch_size, patch_size]
    nH, nW = patches.shape[2], patches.shape[3]

    patches = patches.contiguous().view(B, C, nH * nW, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, p, p]
    return patches, (nH, nW)


def extract_vit_manual_patch_embeddings(model, images, manual_patch_size=64,
                                         vit_input_size=224):
    """
    1. Découpe chaque image en patchs non chevauchants de taille manual_patch_size.
    2. Redimensionne chaque patch à vit_input_size (résolution attendue par le ViT),
       si différente de manual_patch_size.
    3. Fait un forward pass du ViT sur tous les patchs (de toutes les images du
       batch, aplatis ensemble pour profiter du batching), et récupère le
       token [CLS] de chaque patch.
    4. Reforme la séquence de CLS par image : [B, num_patches, C].

    Retourne (embeddings [B, num_patches, C], (B, nH, nW)), au même format
    que les autres fonctions d'extraction patch-level du script.
    """
    B, C, H, W = images.shape

    patches, (nH, nW) = split_into_patches(images, manual_patch_size)
    num_patches = nH * nW

    # Aplatit (batch, patch) ensemble pour un seul forward pass groupé
    patches_flat = patches.reshape(B * num_patches, C, manual_patch_size, manual_patch_size)

    if manual_patch_size != vit_input_size:
        patches_flat = F.interpolate(
            patches_flat, size=(vit_input_size, vit_input_size),
            mode="bilinear", align_corners=False,
        )

    cls_flat = extract_vit_embeddings(model, patches_flat, is_cls=True)  # [B*num_patches, C_emb]

    C_emb = cls_flat.shape[-1]
    embeddings = cls_flat.reshape(B, num_patches, C_emb)  # [B, num_patches, C_emb]

    return embeddings, (B, nH, nW)


# ---------------------------------------------------------------------------
# Pipeline d'extraction + sauvegarde pour un split complet (agnostique du modèle)
# ---------------------------------------------------------------------------

def extract_and_save(dataset, model, extract_fn, output_path, device="cpu",
                      batch_size=16, num_workers=4, **extract_kwargs):
    """
    Parcourt tout le dataset, extrait les embeddings via extract_fn,
    et sauvegarde tout dans un seul fichier .pt.

    extract_fn doit avoir la signature :
        extract_fn(model, images, **kwargs)

    Deux formats de sortie sont supportés :

    - is_cls=True (ViT, mode CLS global) :
        embeddings [N, C]  +  grid_hw = None

    - sinon (ResNet, ViT patch tokens, ou ViT patchs manuels) :
        embeddings [N, num_patches, C]  +  grid_hw = (H, W)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_embeddings = []
    all_labels = []
    all_defect_types = []
    all_paths = []

    is_cls = extract_kwargs.get("is_cls", False)
    grid_hw = None

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]
        defect_types = batch["defect_type"]
        paths = batch["path"]

        if is_cls:
            embeddings = extract_fn(model, images, **extract_kwargs)  # [B, C]
        else:
            embeddings, (B, H, W) = extract_fn(model, images, **extract_kwargs)  # [B, N, C]
            grid_hw = (H, W)

        all_embeddings.append(embeddings.cpu())
        all_labels.extend(labels.tolist())
        all_defect_types.extend(defect_types)
        all_paths.extend(paths)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    payload = {
        "embeddings": embeddings_tensor,
        "labels": torch.tensor(all_labels),
        "defect_types": all_defect_types,
        "paths": all_paths,
        "grid_hw": grid_hw,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    if is_cls:
        print(f"  -> {output_path}  ({embeddings_tensor.shape[0]} images, "
              f"{embeddings_tensor.shape[1]} dims, CLS)")
    else:
        print(f"  -> {output_path}  ({embeddings_tensor.shape[0]} images, "
              f"{embeddings_tensor.shape[1]} patches/image, "
              f"{embeddings_tensor.shape[2]} dims)")


def load_embeddings(path):
    """
    Recharge un fichier d'embeddings sauvegardé par ce script.

    Exemple :
        data = load_embeddings("data/embeddings/wide_resnet50_2/bottle/train.pt")
        embeddings = data["embeddings"]      # [N, H*W, C] ou [N, C] si CLS
        labels = data["labels"]              # [N]
        defect_types = data["defect_types"]  # list[str]
    """
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def main():
    parser = argparse.ArgumentParser(description="Extraction d'embeddings patch-level pour MVTec AD")
    parser.add_argument("--root", type=str, default="data/mv_tec_ad",
                         help="Racine du dataset MVTec AD")
    parser.add_argument("--category", type=str, default="bottle",
                         help="Catégorie MVTec AD, ou 'all' pour toutes les traiter")
    parser.add_argument("--output", type=str, default="data/embeddings",
                         help="Dossier de sortie pour les fichiers .pt")

    parser.add_argument("--model_type", type=str,
                         choices=["resnet", "vit", "vit_manual_patches"],
                         default="resnet",
                         help="Backbone/mode à utiliser pour l'extraction")

    # Arguments spécifiques ResNet
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2",
                         help="[resnet] Nom du modèle timm")
    parser.add_argument("--out_indices", type=int, nargs="+", default=[2, 3],
                         help="[resnet] Indices des couches à extraire (2,3 = layer2,layer3)")
    parser.add_argument("--patch_size", type=int, default=3,
                         help="[resnet] Taille du voisinage pour l'average pooling local")

    # Arguments spécifiques ViT (patch tokens natifs ou CLS global)
    parser.add_argument("--vit_model_name", type=str, default="vit_base_patch16_224",
                         help="[vit / vit_manual_patches] Nom du modèle timm (ex: "
                              "vit_base_patch16_224, vit_small_patch16_224, "
                              "vit_base_patch14_dinov2.lvd142m)")
    parser.add_argument("--is_cls", action="store_true",
                         help="[vit] Récupère l'embedding CLS global au lieu des patch tokens")

    # Arguments spécifiques ViT avec patchs manuels
    parser.add_argument("--manual_patch_size", type=int, default=64,
                         help="[vit_manual_patches] Taille des patchs découpés manuellement "
                              "dans l'image (doit diviser --image_size), sans overlap")
    parser.add_argument("--vit_input_size", type=int, default=224,
                         help="[vit_manual_patches] Résolution vers laquelle chaque patch "
                              "manuel est redimensionné avant d'être passé au ViT")

    parser.add_argument("--image_size", type=int, default=256,
                         help="Taille d'image utilisée pour resnet et vit (le ViT interpole "
                              "son positional embedding si différent de sa résolution native)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    categories = ALL_CATEGORIES if args.category == "all" else [args.category]

    print(f"Device : {args.device}")
    print(f"Model type : {args.model_type}")

    # -----------------------------------------------------------------
    # Construction du modèle + fonction d'extraction + transform + dossier
    # -----------------------------------------------------------------
    if args.model_type == "resnet":
        model = build_resnet_backbone(
            model_name=args.backbone,
            out_indices=tuple(args.out_indices),
            device=args.device,
        )
        extract_fn = extract_resnet_patch_embeddings
        extract_kwargs = {"patch_size": args.patch_size}
        transform = None  # utilise le transform par défaut de MVTecAD (resize + ToTensor)
        model_folder = args.backbone
        print(f"Backbone : {args.backbone}, out_indices={tuple(args.out_indices)}")

    elif args.model_type == "vit":
        model = build_vit_model(
            model_name=args.vit_model_name,
            image_size=args.image_size,
            device=args.device,
        )
        extract_fn = extract_vit_embeddings
        extract_kwargs = {"is_cls": args.is_cls}
        transform = build_vit_transform(model, image_size=args.image_size)
        suffix = "cls" if args.is_cls else "patches"
        model_folder = f"{args.vit_model_name}_{suffix}"
        print(f"ViT model : {args.vit_model_name}, image_size={args.image_size}, "
              f"mode={'CLS global' if args.is_cls else 'patch tokens natifs'}")

    elif args.model_type == "vit_manual_patches":
        # Le ViT est construit à la résolution des patchs manuels (vit_input_size),
        # pas à la résolution de l'image entière : chaque patch lui est présenté
        # séparément, redimensionné à cette taille.
        model = build_vit_model(
            model_name=args.vit_model_name,
            image_size=args.vit_input_size,
            device=args.device,
        )
        extract_fn = extract_vit_manual_patch_embeddings
        extract_kwargs = {
            "manual_patch_size": args.manual_patch_size,
            "vit_input_size": args.vit_input_size,
        }
        # Le dataset garde des images à args.image_size ; c'est le découpage manuel
        # (dans extract_vit_manual_patch_embeddings) qui produit les patchs, donc
        # on utilise un transform simple resize+ToTensor (pas la normalisation ViT
        # standard, car chaque patch sera de toute façon redimensionné puis passé
        # au modèle -- on applique la normalisation ImageNet générique ici).
        transform = T.Compose([
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        model_folder = f"{args.vit_model_name}_manual_p{args.manual_patch_size}"
        print(f"ViT model : {args.vit_model_name}, image_size={args.image_size}, "
              f"manual_patch_size={args.manual_patch_size}, "
              f"vit_input_size={args.vit_input_size} (CLS par patch, sans overlap)")

    else:
        raise ValueError(f"model_type inconnu : {args.model_type}")

    output_dir = Path(args.output)

    for category in categories:
        print(f"\n=== Catégorie : {category} ===")

        train_dataset = MVTecAD(root=args.root, category=category, split="train",
                                 transform=transform, image_size=args.image_size)
        test_dataset = MVTecAD(root=args.root, category=category, split="test",
                                transform=transform, image_size=args.image_size)

        print(f"Train : {len(train_dataset)} images | Test : {len(test_dataset)} images")
        defect_counts = Counter(s[2] for s in test_dataset.samples)
        print(f"Répartition test : {dict(defect_counts)}")

        print("Extraction train...")
        extract_and_save(
            train_dataset, model, extract_fn,
            output_path=output_dir / model_folder / category / "train.pt",
            device=args.device, batch_size=args.batch_size,
            num_workers=args.num_workers, **extract_kwargs,
        )

        print("Extraction test...")
        extract_and_save(
            test_dataset, model, extract_fn,
            output_path=output_dir / model_folder / category / "test.pt",
            device=args.device, batch_size=args.batch_size,
            num_workers=args.num_workers, **extract_kwargs,
        )

    print("\nTerminé.")


if __name__ == "__main__":
    main()