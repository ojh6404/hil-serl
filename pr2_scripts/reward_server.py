#!/usr/bin/env python3
"""
Reward server for RL - REST API server for yes/no VLM-based reward computation
"""

import io
import base64
import json
import argparse
import re

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

app = Flask(__name__)

# Global variables to store model and config
MODEL = None
PROCESSOR = None
ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Reward Server for RL")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen VLM model name or path",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "auto"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: None, uses model default)",
    )

    # Processor arguments
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=None,
        help="Minimum number of pixels for image processing (default: None, uses model default)",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum number of pixels for image processing (default: None, uses model default)",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate (default: 32, enough for yes/no)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic, default: 0.1)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling (vs greedy decoding)",
    )

    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind server to")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

    return parser.parse_args()


def load_image_from_bytes(image_bytes):
    """
    Load PIL image from bytes.

    Args:
        image_bytes: Image bytes (JPEG, PNG, etc.)

    Returns:
        PIL.Image: Loaded image
    """
    image_pil = Image.open(io.BytesIO(image_bytes))
    return image_pil


def parse_yes_no_answer(text: str) -> int:
    """
    Parse yes/no answer from generated text and return reward (0 or 1).

    Args:
        text: Generated text from VLM

    Returns:
        int: 1 if "yes", 0 if "no", -1 if uncertain
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower().strip()

    # Check for explicit yes/no at the start
    if text_lower.startswith("yes"):
        return 1
    elif text_lower.startswith("no"):
        return 0

    # Check for yes/no anywhere in the text (as a word boundary)
    yes_pattern = r'\byes\b'
    no_pattern = r'\bno\b'

    has_yes = bool(re.search(yes_pattern, text_lower))
    has_no = bool(re.search(no_pattern, text_lower))

    if has_yes and not has_no:
        return 1
    elif has_no and not has_yes:
        return 0
    else:
        # Ambiguous or no clear answer
        return -1


@torch.no_grad()
def run_inference(image_pil, prompt, temperature=None, top_p=None, do_sample=None, max_new_tokens=None):
    """
    Run VLM inference and return yes/no reward.

    Args:
        image_pil: PIL Image
        prompt: Question prompt (e.g., "Does the fridge in the scene open? Answer yes or no.")
        temperature: Sampling temperature (optional, uses default if None)
        top_p: Top-p sampling (optional, uses default if None)
        do_sample: Whether to use sampling (optional, uses default if None)
        max_new_tokens: Maximum new tokens (optional, uses default if None)

    Returns:
        dict: {
            "reward": int (1 for yes, 0 for no, -1 for uncertain),
            "answer": str (generated text),
            "confidence": str ("certain" if reward is 0 or 1, "uncertain" if -1)
        }
    """
    global MODEL, PROCESSOR, ARGS

    # Use provided values or fall back to defaults
    temperature = temperature if temperature is not None else ARGS.temperature
    top_p = top_p if top_p is not None else ARGS.top_p
    do_sample = do_sample if do_sample is not None else ARGS.do_sample
    max_new_tokens = max_new_tokens if max_new_tokens is not None else ARGS.max_new_tokens

    # Prepare messages in Qwen format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(ARGS.device)

    # Generate
    generated_ids = MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"Generated answer: {output_text}")

    # Parse yes/no answer
    reward = parse_yes_no_answer(output_text)

    # Determine confidence
    confidence = "certain" if reward in [0, 1] else "uncertain"

    results = {
        "reward": reward,
        "answer": output_text,
        "confidence": confidence,
    }

    return results


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/reward", methods=["POST"])
def reward():
    """
    Reward computation endpoint.

    Expected JSON input:
    {
        "image": base64 encoded image (JPEG, PNG, etc.),
        "prompt": str (e.g., "Does the fridge in the scene open? Answer yes or no."),
        "temperature": float (optional, default: from args),
        "top_p": float (optional, default: from args),
        "do_sample": bool (optional, default: from args),
        "max_new_tokens": int (optional, default: from args)
    }

    Returns:
    {
        "reward": int (1 for yes, 0 for no, -1 for uncertain),
        "answer": str (generated text),
        "confidence": str ("certain" or "uncertain")
    }
    """
    try:
        data = request.get_json()

        # Parse input
        image_b64 = data.get("image")
        prompt = data.get("prompt")

        # Optional parameters
        temperature = data.get("temperature")
        top_p = data.get("top_p")
        do_sample = data.get("do_sample")
        max_new_tokens = data.get("max_new_tokens")

        # Validate required fields
        if not image_b64 or not prompt:
            return jsonify({"error": "Missing required fields: 'image' and 'prompt'"}), 400

        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image_pil = load_image_from_bytes(image_bytes)

        # Run inference
        results = run_inference(
            image_pil,
            prompt,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def initialize_model(args):
    """Initialize VLM model and processor"""
    global MODEL, PROCESSOR, ARGS

    ARGS = args

    print("=" * 80)
    print("Initializing Qwen VLM Model")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.torch_dtype}")
    if args.attn_implementation:
        print(f"Attention: {args.attn_implementation}")

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    torch_dtype = dtype_map[args.torch_dtype]

    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print("\nLoading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    model.eval()
    print("✓ Model loaded successfully")

    # Load processor
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    print("\nLoading processor...")
    if processor_kwargs:
        print(f"  Custom processor settings: {processor_kwargs}")
        processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)
    else:
        processor = AutoProcessor.from_pretrained(args.model_name)
    print("✓ Processor loaded successfully")

    # Print configuration
    print("\nGeneration Configuration:")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Do sample: {args.do_sample}")

    MODEL = model
    PROCESSOR = processor


def main():
    args = parse_args()

    # Initialize model
    initialize_model(args)

    # Start Flask server
    print("\n" + "=" * 80)
    print(f"Starting Reward Server on {args.host}:{args.port}")
    print("=" * 80)
    print("\nEndpoints:")
    print(f"  - Health check: http://{args.host}:{args.port}/health")
    print(f"  - Reward computation: http://{args.host}:{args.port}/reward (POST)")
    print("\n")
    print("Example usage:")
    print('  curl -X POST http://localhost:5001/reward \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"image": "<base64_encoded_image>", "prompt": "Does the fridge in the scene open? Answer yes or no."}\'')
    print("\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
