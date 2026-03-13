import os
import io
import sys
import time
import torch
import json
from openai import OpenAI
import base64
from PIL import Image
import healpy as hp
import trimesh
import open_clip
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch.nn.functional as F
import traceback
from viewpoint_selection import get_optimal_viewpoint_and_image


def write_valid_num_to_txt(file_path, valid_i):
    with open(file_path, 'w') as file:
        file.write(str(valid_i))


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


@torch.no_grad()
def extract_text_feat(texts, clip_model, ):
    text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
    return clip_model.encode_text(text_tokens)


def load_npy(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    xyz = data['xyz']
    xyz[:, [1, 2]] = xyz[:, [2, 1]]
    rgb = data['rgb']
    return xyz, rgb


def load_batch(directory, npy_lists):
    XYZ = []
    RGB = []
    file_paths = [os.path.join(directory, npy) for npy in npy_lists]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: load_npy(x), file_paths))
    for i, (xyz, rgb) in enumerate(results):
        XYZ.append(xyz)
        RGB.append(rgb)
    XYZ = torch.from_numpy(np.concatenate(XYZ))
    RGB = torch.from_numpy(np.concatenate(RGB))
    XYZ = XYZ.view(-1, 10000, 3).float().to(device='cuda')
    RGB = RGB.view(-1, 10000, 3).float().to(device='cuda')
    return XYZ, RGB


def extract_shape_feats(XYZ, RGB, shape_encoder, backbone):
    Feat = torch.cat((XYZ, RGB), dim=2)
    shape_feats = []
    if backbone == 'PointBERT':
        with torch.no_grad():
            for k in range(0, Feat.shape[0], 10):
                shape_feat = shape_encoder(XYZ[k:k + 10], Feat[k:k + 10])
                shape_feats.append(shape_feat)
    else:
        print('Please specify the correct backbone type!')
    shape_feats = torch.cat(shape_feats, dim=0)
    return shape_feats


if __name__ == '__main__':
    """
    1. Load pre-trained MEM model and OpenCLIP
    """
    print("Loading MEM model...")
    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14',
                                                                                     pretrained='laion2b_s39b_b160k',
                                                                                     cache_dir="../models/Pretrained_models/open_clip_model/")
    open_clip_model.cuda().eval()
    """
    2. Load 3D model library -> Load pre-extracted features locally
    """
    dir_glbs = '/home/l20/.objaverse/hf-objaverse-v1/glbs/000-000'
    sorted_glbs = sorted(os.listdir(dir_glbs))
    # Load pre-extracted features from local file
    shape_feats = torch.load('../data/shape_feats.pt')
    """
    3. Initialize 2D-MLLM API
    """
    client = OpenAI(
        api_key="your_api",
        base_url="https://xiaoai.plus/v1"
    )

    """
    4. Process each query one by one
    """
    query_texts = []
    # Open and read the JSON file
    with open("../data/3dmodel_query_matches.json", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for item in json_data:
        if isinstance(item, dict) and 'query' in item:
            query_text = item['query']
            if query_text.strip():
                query_texts.append(query_text)

    Pre_matches = []
    start_time = time.time()
    for i in range(len(query_texts)):
        cur_query = query_texts[i]
        print(f"Processing {i}th query text：", cur_query)
        # Extract text features
        query_feat = extract_text_feat([cur_query], open_clip_model)
        # Re-ranking
        Sim_t_shape = F.cosine_similarity(F.normalize(query_feat, dim=1),
                                          F.normalize(shape_feats, dim=1),
                                          dim=1)
        reranked_idxes = torch.argsort(Sim_t_shape, descending=True)
        tmp_matches = []
        count = 0
        for j in range(len(reranked_idxes)):
            print(f"Processing {j}th 3D-model")
            scene = trimesh.load(os.path.join(dir_glbs, sorted_glbs[reranked_idxes[j]]))
            model_uid = sorted_glbs[reranked_idxes[j]][:-4]
            selected_image, Base_images, All_images, opt_viewpoint_idx = get_optimal_viewpoint_and_image(
                scene=scene,
                model_uid=model_uid,
                query_feat=query_feat,
                open_clip_model=open_clip_model,
                open_clip_preprocess=open_clip_preprocess
            )
            output_buffer = io.BytesIO()
            selected_image.save(output_buffer, format="PNG")
            selected_img_data = output_buffer.getvalue()
            # Call 2D-MMLM API to judge whether it matches
            base64_image = base64.b64encode(selected_img_data).decode('utf-8')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": 'Query: "' + cur_query + '"\n'
                                                             'Does the object shown in the image match the given query? '
                                                             'Please base your judgment on the object’s category, as well as '
                                                             'key characteristics such as shape, color, and specific details. '
                                                             'If you are certain it matches, reply "1". '
                                                             'If you are certain it does not match, reply "0". '
                                                             'If the current view does not provide enough information to make a confident '
                                                             'and definitive judgment, and additional views are required for confirmation, reply "2".'
                        }
                    ]
                }
            ]
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300
            )
            response = completion.choices[0].message.content.strip()
            print('Main judgment result:', response)

            # Keep all logic below (handle API response, assisted judgment, count increment, etc.)
            if response == '1':
                tmp_matches.append(sorted_glbs[reranked_idxes[j]][:-4])
                count = 0
            elif response == '0':
                count += 1
            elif response == '2':
                Base_images_ = [x for i, x in enumerate(Base_images) if i != opt_viewpoint_idx]
                for q in range(len(Base_images_)):
                    added_image = Base_images_[q]
                    output_buffer = io.BytesIO()
                    added_image.save(output_buffer, format="PNG")
                    added_img_data = output_buffer.getvalue()
                    added_base64_image = base64.b64encode(added_img_data).decode('utf-8')
                    assistant_message = completion.choices[0].message.model_dump()
                    messages.append(assistant_message)
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{added_base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": 'This is another view of the object. Can it help '
                                        'you further confirm the judgment? If yes, reply directly with the '
                                        'matching result ("1" or "0"). If no, reply with "2".'
                            }
                        ]
                    })
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=300
                    )
                    response = completion.choices[0].message.content.strip()
                    print('Assisted judgment result:', response)
                    if response == '1':
                        tmp_matches.append(sorted_glbs[reranked_idxes[j]][:-4])
                        count = 0
                        break
                    elif response == '0':
                        count += 1
                        break
                    elif response == '2':
                        if q != len(Base_images_) - 1:
                            continue
                        else:
                            count += 1
                    else:
                        raise ValueError(f"Invalid assisted judgment: {response} (expected '1' or '0')")
            else:
                raise ValueError(f"Invalid main judgment: {response} (expected '1', '0', or '2')")

            if count >= 5:
                break

        Pre_matches.append(tmp_matches)
        with open(f'Pre_matches.json', "w") as f:
            json.dump(Pre_matches, f)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")