import os
import io
import numpy as np
import torch
import torch.nn.functional as F
import healpy as hp
import trimesh
from PIL import Image
import open_clip

# Reused sorting function from original code (dependency)
def sort_key(filename):
    if 'base' in filename:
        parts = filename.replace('base', '0').split('_')
    else:
        parts = filename.split('_')
    i = int(parts[1])
    j = int(parts[2].replace('.png', ''))
    return (i, j)

def sort_png_files(folder_path):
    all_files = os.listdir(folder_path)
    png_files = [f for f in all_files if f.endswith('.png')]
    sorted_files = sorted(png_files, key=sort_key)
    return sorted_files

@torch.no_grad()
def extract_image_feat(images, clip_model, clip_preprocess):
    """Extract image features (reused original logic)"""
    image_tensors = [clip_preprocess(image) for image in images]
    image_tensors = torch.stack(image_tensors, dim=0).float().cuda()
    image_features = clip_model.encode_image(image_tensors)
    image_features = image_features.reshape((-1, image_features.shape[-1]))
    return image_features

def get_optimal_viewpoint_and_image(
    scene,
    model_uid,
    query_feat,
    open_clip_model,
    open_clip_preprocess
):
    """
    Compute the optimal viewpoint for a 3D model and return the corresponding image
    Parameters:
    - scene: 3D model scene loaded by trimesh
    - model_uid: Unique identifier of the 3D model (without .glb suffix)
    - query_feat: Feature vector of the query text (1, 1280)
    - open_clip_model: Preloaded OpenCLIP model
    - open_clip_preprocess: Image preprocessing function for OpenCLIP
    Returns:
    - selected_image: Optimal image from the best viewpoint (PIL.Image)
    - Base_images: List of base view images
    - All_images: List of all rendered images
    - opt_viewpoint_idx: Index of the optimal viewpoint
    """
    # 1. Compute model center and radius
    try:
        bounding_sphere = scene.bounding_sphere
        center = bounding_sphere.primitive.center
        sphere_radius = bounding_sphere.primitive.radius
    except Exception as e:
        print(f"Failed to compute bounding sphere, use bounding box instead! Error: {e}")
        min_bound, max_bound = scene.bounds
        center = (min_bound + max_bound) / 2
        sphere_radius = (np.linalg.norm(max_bound - min_bound)) / 2

    # 2. Compute optimal camera distance from model center
    camera_fov = np.pi / 4
    opt_d = sphere_radius / np.sin(camera_fov / 2)

    # 3. Uniform sampling on sphere using HEALPix
    N_side = 1
    N_points = hp.nside2npix(N_side)
    theta, phi = hp.pix2ang(N_side, np.arange(N_points))
    x = center[0] + opt_d * np.sin(theta) * np.cos(phi)
    y = center[1] + opt_d * np.sin(theta) * np.sin(phi)
    z = center[2] + opt_d * np.cos(theta)
    points = np.vstack((x, y, z)).T

    # 4. Compute z-axis direction for each sampling point
    z_axis = points - center
    z_axis = z_axis / np.linalg.norm(z_axis, axis=1, keepdims=True)

    # 5. Initialize reference vector for x-axis
    random_vectors = np.random.rand(z_axis.shape[0], 3) * 2 - 1
    random_vectors = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
    dot_product = np.abs(np.einsum('ij,ij->i', z_axis, random_vectors))
    x_axis_ref = np.where(
        dot_product[:, np.newaxis] < 0.99,
        random_vectors,
        np.array([1, 0, 0])[np.newaxis, :]
    )

    # 6. Compute orthogonal x/y axes using Gram-Schmidt
    x_axis = x_axis_ref - np.einsum('ij,ij->i', x_axis_ref, z_axis)[:, np.newaxis] * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)

    # 7. Compute camera pose matrices
    rotation_matrices = np.stack((x_axis, y_axis, z_axis), axis=-1)
    translation_matrices = points[:, :, np.newaxis]
    upper_part = np.concatenate((rotation_matrices, translation_matrices), axis=2)
    bottom_row = np.tile(np.array([0, 0, 0, 1]), (len(rotation_matrices), 1)).reshape(
        len(rotation_matrices), 1, 4)
    pose_matrices = np.concatenate((upper_part, bottom_row), axis=1)

    # 8. Load and render images
    Base_images = []
    All_images = []
    resolution = (800, 600)
    for k in range(len(pose_matrices)):
        scene.camera_transform = pose_matrices[k]
        img_data = scene.save_image(resolution=resolution)
        image = Image.open(trimesh.util.wrap_as_stream(img_data)).convert("RGB")
        Base_images.append(image)
        All_images.append(image)
        angles = range(30, 360, 30)  # 30, 60, ..., 330
        for angle in angles:
            rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
            All_images.append(rotated_image)
    image_feats = extract_image_feat(All_images, open_clip_model, open_clip_preprocess)

    # 9. Compute semantic similarity between images and query text
    tmp_Sim_t_img = F.cosine_similarity(F.normalize(query_feat, dim=1),
                                        F.normalize(image_feats, dim=1),
                                        dim=1)
    Sim_t_img = tmp_Sim_t_img.reshape(-1, 1 + len(angles))
    sim_for_viewpoint = Sim_t_img.mean(dim=1)

    # 10. Select optimal viewpoint and corresponding image
    opt_viewpoint_idx = torch.argmax(sim_for_viewpoint).item()
    opt_img_idx = torch.argmax(Sim_t_img[opt_viewpoint_idx]).item()
    selected_image = All_images[opt_viewpoint_idx * (1 + len(angles)) + opt_img_idx]

    return selected_image, Base_images, All_images, opt_viewpoint_idx