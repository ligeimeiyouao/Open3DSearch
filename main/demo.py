import os
import io
import sys
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

@torch.no_grad()
def extract_text_feat(texts, clip_model, ):
    text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
    return clip_model.encode_text(text_tokens)

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
    4. Process sample query 
    """

    sample_query = "a sword with a metal handle"
    print(f"Processing query text：", sample_query)

    # Extract text features
    query_feat = extract_text_feat([sample_query], open_clip_model)
    # Re-ranking
    Sim_t_shape = F.cosine_similarity(F.normalize(query_feat, dim=1),
                                      F.normalize(shape_feats, dim=1),
                                      dim=1)
    reranked_idxes = torch.argsort(Sim_t_shape, descending=True)
    Pre_matches = []
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
                        "text": 'Query: "' + sample_query + '"\n'
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
            Pre_matches.append(sorted_glbs[reranked_idxes[j]][:-4])
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
                    Pre_matches.append(sorted_glbs[reranked_idxes[j]][:-4])
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

    print(Pre_matches)
    # If you want to visualize the predicted matching models, you can run the following code.
    for uid in Pre_matches:
        pre_scene = trimesh.load(os.path.join(dir_glbs, f"{uid}.glb"))
        pre_scene.show()