import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .face_detector import FaceDetector
from .dataset import get_eval_transforms
from .model import build_efficientnet_b3, load_model_weights

CLASS_NAMES = ["fake", "real"]


class DeepFakePredictor:
    """
    Production-ready inference pipeline for classifying facial images as Authentic or AI-Generated.
    Includes automated face detection, cropping, and probability calibration.
    """
    def __init__(self, model_path="best_model_rebuilt", device=None, threshold=0.5):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.threshold = threshold
        self.transform = get_eval_transforms()
        self.face_detector = FaceDetector(device=self.device)

        # Build and load model
        self.model = build_efficientnet_b3(num_classes=2, pretrained=False)
        self.model = load_model_weights(self.model, model_path, self.device)
        self.model.eval()

    def predict(self, image_input, threshold=None, detect_face=True) -> dict:
        """
        Classifies an image as Authentic (Real) or AI-Generated (Fake).

        Args:
            image_input (str or PIL.Image): Path to image file or PIL Image object.
            threshold (float, optional): Custom probability cutoff for Authentic classification.
            detect_face (bool): Whether to perform MTCNN face detection and cropping.

        Returns:
            dict containing prediction results, probabilities, and face detection metadata.
        """
        cutoff = threshold if threshold is not None else self.threshold

        image_path = None
        if isinstance(image_input, str):
            image_path = image_input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        face_detected = False
        bbox = None
        cropped_image = image

        if detect_face:
            cropped_image, face_detected, bbox = self.face_detector.detect_and_crop(image)

        input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        fake_prob = float(probs[0].item())
        real_prob = float(probs[1].item())

        # Apply threshold (0 = fake, 1 = real)
        if real_prob >= cutoff:
            predicted_label = "Authentic"
            label_idx = 1
            confidence = real_prob * 100.0
        else:
            predicted_label = "AI-Generated"
            label_idx = 0
            confidence = fake_prob * 100.0

        return {
            "image_path": image_path,
            "predicted_class": predicted_label,
            "label_index": label_idx,
            "fake_probability": fake_prob * 100.0,
            "real_probability": real_prob * 100.0,
            "confidence_percentage": confidence,
            "threshold_used": cutoff,
            "face_detected": face_detected,
            "bbox": bbox
        }


def predict_single_image(image_path, model_path="best_model_rebuilt", device=None, threshold=0.5, show=False):
    """
    Convenience function to classify a single local image file.
    """
    predictor = DeepFakePredictor(model_path=model_path, device=device, threshold=threshold)
    result = predictor.predict(image_path, threshold=threshold)

    print("\n" + "=" * 55)
    print(f" Image Analysis: {os.path.basename(image_path)}")
    print("=" * 55)
    print(f" Verdict:              {result['predicted_class'].upper()}")
    print(f" Confidence:           {result['confidence_percentage']:.2f}%")
    print(f" AI-Generated (Fake):  {result['fake_probability']:.2f}%")
    print(f" Authentic (Real):     {result['real_probability']:.2f}%")
    print(f" Face Detected:        {result['face_detected']} {result['bbox'] if result['bbox'] else ''}")
    print("=" * 55 + "\n")

    if show and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Verdict: {result['predicted_class']}\n"
            f"Fake: {result['fake_probability']:.1f}% | Real: {result['real_probability']:.1f}%",
            fontsize=12, fontweight="bold"
        )
        plt.show()

    return result
