from openai import OpenAI
from os import getenv


import zipfile
import os
import base64
import requests
import csv
import json
import cv2


# class label list

# 'voc': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 'cityscapes': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'],

# 'ade20k': ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

# 'coco': ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


caption_system_prompt = '''
You are an expert vision-language assistant trained in detailed scene understanding and semantic segmentation. When given an image, generate a comprehensive and structured caption that:

1. Identify and describe all visible objects, regions, and surfaces based on the following predefined class labels.
2. Describes each object with fine-grained attributes (e.g., color, size, material, texture, state).
3. Uses spatial terms to locate objects (e.g., "in the foreground", "to the left", "in the top-right corner").
4. Groups semantically similar regions (e.g., "a group of people", "rows of trees").
5. Uses consistent class names based on semantic segmentation class labels.
6. Avoids hallucinations or guesses—only describe what is visually present.

Class labels:
['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

Your outputs should be factual, richly descriptive, and useful for vision-language tasks, training data augmentation, or multi-modal captioning. 
Directly describe with brevity and as brief as possible the scene or characters without any introductory phrase like 'This image shows', 'In the scene', 'This image depicts' or similar phrases. Just start describing the scene please.
'''



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="your api key"
    )


def get_caption(url):
    # caption_prompt = "Directly describe with brevity and as brief as possible the scene or characters without any introductory phrase like 'This image shows', 'In the scene', 'This image depicts' or similar phrases. Just start describing the scene please."

    completion = client.chat.completions.create(
    extra_headers={
        #"HTTP-Referer": $YOUR_SITE_URL, # Optional, for including your app on openrouter.ai rankings.
        #"X-Title": $YOUR_APP_NAME, # Optional. Shows in rankings on openrouter.ai.
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Test",
    },
    
    extra_body={},
    model="google/gemma-3-27b-it",
    # model="openai/gpt-4o-2024-08-06",
    # model="google/gemma-3-27b-it:free",
    # model="google/gemini-2.0-flash-exp:free",
    # model="qwen/qwen2.5-vl-72b-instruct:free",
    # model="google/gemini-2.5-flash",
    # model="openai/gpt-4.1-mini",
    messages=[
        # {
        #     "role": "system",
        #     "content": caption_system_prompt
        # },
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": caption_system_prompt # caption_prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": url
                }
                }
            ]
        }
    ]
    )

    return completion.choices[0].message.content


data = []

folder = "data/coco/"  # "data/ADE20K/ADEChallengeData2016/" "data/cityscapes/" "data/voc/VOCdevkit/VOC2012/"

filenames = []

# get list of query file paths
# with open("cityscapes.txt", "r") as f:
# with open("voc.txt", "r") as f:
# with open("ade20k.txt", "r") as f:
with open("coco.txt", "r") as f:  
    lines = f.readlines()
    for line in lines:
        filename = os.path.join(folder, line.strip().split(" ")[0])
        filenames.append(filename)
        
        
for image_path in filenames:   
    json_path = os.path.join(os.path.join(folder, "json_gemma3"), os.path.basename(image_path)[:-4]+".json")
    
    # print(image_path, json_path)
    if os.path.exists(json_path):
        continue
    
    base64_image = encode_image(image_path)
    
    ext = os.path.splitext(image_path)[-1].lower()

    if ext in [".jpg", ".jpeg"]:
        url = f"data:image/jpeg;base64,{base64_image}"
    elif ext in [".png"]:
        url = f"data:image/png;base64,{base64_image}"  
    else:
        url = f"data:image/{ext[1:]};base64,{base64_image}"
        # print(image_path, ext)
     
    caption = get_caption(url)
    print(caption)

    # data.append({"file_name": image_path, "text": caption})
    data = {"file_name": image_path, "text": caption}
    
    with open(json_path, "w") as f:
        f.write(json.dumps(data))
        