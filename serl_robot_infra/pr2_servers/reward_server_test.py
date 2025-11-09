#!/usr/bin/env python3

import cv2
import numpy as np
import requests
import base64
import json
import time
from typing import Tuple

def pil_to_base64(img_array: np.ndarray) -> str:
    """
    Convert numpy array to base64 string
    Args:
        img_array (np.ndarray): Image array
    Returns:
        str: Base64 string
    """
    return base64.b64encode(img_array.tobytes()).decode("utf-8")

def process_video(video_path: str, server_url: str = "http://localhost:5004/segment") -> None:
    """
    Process video frame by frame and send to reward server
    Args:
        video_path (str): Path to video file
        server_url (str): URL of the reward server
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Prepare request data
            data = {
                "img": pil_to_base64(frame_rgb),
                "shape": list(frame_rgb.shape)
            }

            # Send request to server
            try:
                response = requests.post(server_url, json=data)
                response.raise_for_status()

                # Process response
                result = response.json()
                print(f"Frame {frame_count}: Open probability = {result['prob']:.3f}")

            except requests.exceptions.RequestException as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

            frame_count += 1

    finally:
        cap.release()
        print(f"\nProcessed {frame_count} frames")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video frames with reward server")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--server_url", type=str, default="http://localhost:5004/segment",
                        help="URL of the reward server")

    args = parser.parse_args()

    process_video(args.video_path, args.server_url)
