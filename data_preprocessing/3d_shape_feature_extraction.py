import os
import re
import sys
import torch
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath("../"))

import models
from param import parse_args
from utils.misc import load_config
from huggingface_hub import hf_hub_download

# -------------------------- Core Configuration (Adjustable) --------------------------
BACKBONE = 'PointBERT'  # Options: 'PointBERT' / 'SparseConv'
PRETRAINED_MODEL_PATH = '../models/Pretrained_models/openshape_model/model.pt'
CONFIG_PATH = "../configs/train.yaml"
NPY_DIR = '../data/npys'  # Directory for .npy files
SAVE_FEAT_PATH = '../data/shape_feats.pt'  # Path to save extracted features
HF_MODEL_REPO = "OpenShape/openshape-pointbert-vitg14-rgb"
HF_MODEL_FILENAME = "model.pt"

# -------------------------- Core Functions (Fully Preserved) --------------------------
def load_3d_model(config, backbone):
    # Initialize model and move to GPU
    model = models.make(config).cuda()

    model_dir = os.path.dirname(PRETRAINED_MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Local model not found, auto downloading from Hugging Face: {HF_MODEL_REPO}")
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILENAME,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to: {PRETRAINED_MODEL_PATH}")

    # Load pretrained weights (handle 'module.' prefix)
    checkpoint = torch.load(PRETRAINED_MODEL_PATH)
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    model.eval()  # Evaluation mode

    return model


def load_npy(file_path):
    """Load a single npy file, handle axis swapping (for Objaverse dataset compatibility)"""
    data = np.load(file_path, allow_pickle=True).item()
    xyz = data['xyz']
    xyz[:, [1, 2]] = xyz[:, [2, 1]]  # swap y and z axis (Objaverse shape adaptation)
    rgb = data['rgb']
    return xyz, rgb


def load_batch(directory, npy_lists):
    """Load npy files in batch, accelerated with multi-threading"""
    XYZ = []
    RGB = []

    # Construct full file paths
    file_paths = [os.path.join(directory, npy) for npy in npy_lists]

    # Multi-threaded npy loading
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_npy, file_paths))

    # Concatenate data and convert to Tensor
    for xyz, rgb in results:
        XYZ.append(xyz)
        RGB.append(rgb)

    XYZ = torch.from_numpy(np.concatenate(XYZ))
    RGB = torch.from_numpy(np.concatenate(RGB))

    # Reshape + move to GPU: (batch_size, 10000, 3)
    XYZ = XYZ.view(-1, 10000, 3).float().to(device='cuda')
    RGB = RGB.view(-1, 10000, 3).float().to(device='cuda')

    return XYZ, RGB


@torch.no_grad()  # Disable gradient computation to save GPU memory
def extract_shape_feats(XYZ, RGB, shape_encoder, backbone):
    """Extract 3D shape features"""
    # Concatenate coordinate and color features
    Feat = torch.cat((XYZ, RGB), dim=2)
    shape_feats = []

    if backbone == 'PointBERT':
        # Process in batches (avoid GPU memory overflow)
        for k in range(0, Feat.shape[0], 10):
            shape_feat = shape_encoder(XYZ[k:k + 10], Feat[k:k + 10])  # (batch, 1280)
            shape_feats.append(shape_feat)
    elif backbone == 'SparseConv':
        # SparseConv branch (not implemented in original code, can be added as needed)
        raise NotImplementedError("SparseConv branch needs implementation")
    else:
        raise ValueError(f"Please specify the correct backbone type: {backbone}")

    # Concatenate features from all batches
    shape_feats = torch.cat(shape_feats, dim=0)
    return shape_feats


# -------------------------- Main Execution Logic (Direct Callable) --------------------------
def extract_3d_features():
    """One-click 3D model feature extraction and saving"""
    # 1. Load configuration + 3D model encoder
    print("Loading OpenShape model...")
    cli_args, extras = parse_args(sys.argv[1:])
    config = load_config(CONFIG_PATH, cli_args=vars(cli_args), extra_args=extras)
    open_shape_model = load_3d_model(config, BACKBONE)

    # 2. Load npy file list (sorted by name)
    sorted_npys = sorted(os.listdir(NPY_DIR))
    print(f"Found {len(sorted_npys)} npy files in {NPY_DIR}")

    # 3. Load XYZ+RGB data in batch
    XYZ, RGB = load_batch(NPY_DIR, sorted_npys)
    print(f"Loaded XYZ shape: {XYZ.shape}, RGB shape: {RGB.shape}")

    # 4. Extract features
    shape_feats = extract_shape_feats(XYZ, RGB, open_shape_model, backbone=BACKBONE)
    print(f"Extracted shape features shape: {shape_feats.shape}")

    # 5. Save features locally
    torch.save(shape_feats, SAVE_FEAT_PATH)
    print(f"Shape features saved to {SAVE_FEAT_PATH}")

    return shape_feats


# -------------------------- Load Pre-extracted Features (Helper Function) --------------------------
def load_extracted_3d_features(feat_path=SAVE_FEAT_PATH):
    """Load pre-extracted 3D shape features from local file"""
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    shape_feats = torch.load(feat_path)
    print(f"Loaded pre-extracted features shape: {shape_feats.shape}")
    return shape_feats


if __name__ == '__main__':
    # Run feature extraction
    extract_3d_features()

    # (Optional) Load pre-extracted features
    # shape_feats = load_extracted_3d_features()