import io
import random
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class JPEGCompression:
    """
    Simulates JPEG compression artifacts by re-encoding images at varying quality factors.
    This prevents the CNN from relying on dataset-specific JPEG/PNG compression artifacts.
    """
    def __init__(self, quality_range=(40, 95), p=0.7):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomGaussianBlur:
    """Applies random Gaussian blur to mimic low-resolution / web image uploads."""
    def __init__(self, radius_range=(0.1, 2.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_range[0], self.radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def get_train_transforms(img_size=224):
    """
    Robust training transforms with quality normalization & heavy augmentations.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        JPEGCompression(quality_range=(40, 95), p=0.7),
        RandomGaussianBlur(radius_range=(0.1, 1.5), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transforms(img_size=224):
    """
    Standard evaluation / inference transforms.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class PILListDataset(Dataset):
    """
    PyTorch Dataset wrapper for lists of PIL images or dictionary records.
    Supports optional face detector preprocessing.
    """
    def __init__(self, samples, transform=None, face_detector=None):
        """
        Args:
            samples (list): List of dicts containing 'image' and 'label', or tuples (img_path, label).
            transform (callable): Image transforms.
            face_detector (FaceDetector, optional): MTCNN face detector to crop faces prior to transform.
        """
        self.samples = samples
        self.transform = transform
        self.face_detector = face_detector

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        if isinstance(item, dict):
            image = item["image"]
            label = int(item["label"])
        elif isinstance(item, (tuple, list)):
            img_path, label = item
            image = Image.open(img_path).convert("RGB")
            label = int(label)
        else:
            raise TypeError(f"Unsupported sample format: {type(item)}")

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        if self.face_detector is not None:
            image, _, _ = self.face_detector.detect_and_crop(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
