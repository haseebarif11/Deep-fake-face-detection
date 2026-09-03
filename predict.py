#!/usr/bin/env python3
"""
DeepFake Face Detection - Standalone Inference CLI Tool
Classifies facial images as Authentic or AI-Generated.
"""

import os
import sys
import argparse
from glob import glob

# Ensure repo root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import DeepFakePredictor


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Face Detection - Standalone CLI Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to a single image file to analyze."
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Path to a directory containing images to analyze in bulk."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best_model_rebuilt",
        help="Path to saved model weights or folder checkpoint."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Probability threshold cutoff for Authentic classification."
    )
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Disable MTCNN face detection/cropping."
    )

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        print("\nError: Please specify either --image or --dir")
        sys.exit(1)

    # Initialize predictor
    print(f"Loading DeepFake model from: {args.model}...")
    predictor = DeepFakePredictor(model_path=args.model, threshold=args.threshold)
    detect_face = not args.no_face_detect

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image path does not exist: {args.image}")
            sys.exit(1)

        result = predictor.predict(args.image, detect_face=detect_face)
        print("\n" + "=" * 60)
        print(f" File:                 {os.path.basename(result['image_path'])}")
        print(f" Verdict:              {result['predicted_class'].upper()}")
        print(f" Confidence:           {result['confidence_percentage']:.2f}%")
        print(f" AI-Generated (Fake):  {result['fake_probability']:.2f}%")
        print(f" Authentic (Real):     {result['real_probability']:.2f}%")
        print(f" Face Detected:        {result['face_detected']} {result['bbox'] if result['bbox'] else ''}")
        print("=" * 60)

    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory path does not exist: {args.dir}")
            sys.exit(1)

        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob(os.path.join(args.dir, ext)))
            image_paths.extend(glob(os.path.join(args.dir, "**", ext), recursive=True))

        image_paths = sorted(list(set(image_paths)))
        if not image_paths:
            print(f"No valid images found in directory: {args.dir}")
            sys.exit(0)

        print(f"\nAnalyzing {len(image_paths)} images in {args.dir}...\n")
        print(f"{'Image File':<35} | {'Verdict':<15} | {'Confidence':<12} | {'Fake %':<8} | {'Real %':<8}")
        print("-" * 88)

        for p in image_paths:
            try:
                res = predictor.predict(p, detect_face=detect_face)
                filename = os.path.basename(p)
                if len(filename) > 33:
                    filename = filename[:30] + "..."
                print(f"{filename:<35} | {res['predicted_class']:<15} | {res['confidence_percentage']:.2f}%       | {res['fake_probability']:.1f}%    | {res['real_probability']:.1f}%")
            except Exception as e:
                print(f"{os.path.basename(p):<35} | ERROR ({e})")

        print("-" * 88)


if __name__ == "__main__":
    main()
