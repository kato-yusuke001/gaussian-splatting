from PIL import Image
from lang_sam import LangSAM

import os
import numpy as np
import cv2
import pathlib
from tqdm import tqdm

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image



input_dir = "data/coffee2/input"
input_list = list(pathlib.Path(input_dir).glob('**/*.jpg'))

model = LangSAM("vit_b")
text_prompt = "coffee"

mask_output_path = "data/coffee2_masked/images"
if not os.path.exists(mask_output_path):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs(mask_output_path)

for i in tqdm(range(len(input_list))):
    img_file_name = str(input_list[i])
    image_pil = Image.open(img_file_name).convert("RGB")
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    image_cv2 = pil2cv(image_pil)
    if len(masks) > 0:
        final_mask = masks[0]
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = image_cv2[final_mask, :]
        cv2.imwrite(os.path.join(mask_output_path, os.path.basename(img_file_name)), mask_colors)
    else:
        mask_colors = np.zeros((image_cv2.shape[0], image_cv2.shape[1], 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(mask_output_path, os.path.basename(img_file_name)), mask_colors)
    
