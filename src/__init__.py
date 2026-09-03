"""
DeepFake Face Detection Package
"""

from .face_detector import FaceDetector
from .dataset import (
    JPEGCompression,
    get_train_transforms,
    get_eval_transforms,
    PILListDataset
)
from .model import build_efficientnet_b3, load_model_weights
from .inference import predict_single_image, DeepFakePredictor

# Training utilities (only available when sklearn/tqdm are installed)
try:
    from .train import train_one_epoch, validate_one_epoch, evaluate_classifier
except ImportError:
    pass

__all__ = [
    "FaceDetector",
    "JPEGCompression",
    "get_train_transforms",
    "get_eval_transforms",
    "PILListDataset",
    "build_efficientnet_b3",
    "load_model_weights",
    "predict_single_image",
    "DeepFakePredictor",
    "train_one_epoch",
    "validate_one_epoch",
    "evaluate_classifier"
]
