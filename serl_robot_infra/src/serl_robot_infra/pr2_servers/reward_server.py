"""
Reward server
"""

import json
import os
import cv2
import torch
import numpy as np
import requests
import argparse
import base64
from typing import Dict, Optional, Tuple
from PIL import Image
from flask import Flask, request, jsonify

from transformers import CLIPProcessor, CLIPModel

# 메인 실행 코드
device = "cuda"
torch_dtype = torch.float16

# model_name = "openai/clip-vit-base-patch32"
# model_name = "openai/clip-vit-large-patch14"
model_name = "openai/clip-vit-large-patch14-336"

# CLIP 모델 및 프로세서 로드
model = CLIPModel.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    device_map=device,
    torch_dtype=torch_dtype,
)
processor = CLIPProcessor.from_pretrained(model_name)

app = Flask(__name__)


def pil_to_base64(img: Image.Image) -> str:
    """
    Convert PIL Image to base64 string

    Args:
        img (Image.Image): PIL Image

    Returns:
        str: Base64 string
    """
    img_encoded = base64.b64encode(np.array(img).tobytes()).decode("utf-8")
    return img_encoded


def base64_to_pil(data: str, shape: Tuple[int, int]) -> Image.Image:
    """
    Convert base64 string to PIL Image

    Args:
        data (str): Base64 string

    Returns:
        Image.Image: PIL Image
    """
    img = Image.fromarray(np.frombuffer(base64.b64decode(data), np.uint8).reshape(shape))
    return img



def validate_parameters(data: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Validate the input parameters and files for the retrieval pipeline

    Args:
        data (Dict): JSON data from the request

    Returns:
        Tuple[Optional[Dict], Optional[str]]: Tuple containing the parameters and error
    """
    params = {}

    # Validate required parameters
    if "img" not in data:
        return None, "Missing required parameter: src_img"
    if "shape" not in data:
        return None, "Missing required parameter: src_shape"

    # Update required parameters
    params["img"] = base64_to_pil(data["img"], tuple(data["shape"]))
    return params, None


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy"})

@app.route("/segment", methods=["POST"])
def segment_image():
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate parameters
        params, error = validate_parameters(data)
        if error:
            return jsonify({"error": error}), 400

        # CLIP 입력 준비
        inputs = processor(
            text=["A photo of an open fridge door", "A photo of a closed fridge door"],
            images=params["img"],
            return_tensors="pt",
            padding=True
        )

        # 디바이스로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            with torch.autocast(device):
                outputs = model(**inputs)

        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)

        open_prob = probs[0][0].cpu().numpy()
        close_prob = probs[0][1].cpu().numpy()

        print(f"Open probability: {open_prob:.2f}")
        print(f"Close probability: {close_prob:.2f}")


        # Convert results to serializable format
        response = {
            "prob": open_prob,
            "status": "success",
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flask API for segmentation pipeline")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument("--port", type=int, default=5004, help="Port to run the API on")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port)
