import torch
from PIL import Image
import numpy as np

try:
    from facenet_pytorch import MTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


class FaceDetector:
    """
    Face detection and alignment module using MTCNN (facenet-pytorch).
    Falls back gracefully to center crop if no face is detected or if facenet-pytorch is missing.
    """
    def __init__(self, image_size=224, margin=20, device=None):
        self.image_size = image_size
        self.margin = margin
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.mtcnn = None
        if FACENET_AVAILABLE:
            try:
                self.mtcnn = MTCNN(
                    keep_all=False,
                    select_largest=True,
                    post_process=False,
                    device=self.device
                )
            except Exception as e:
                print(f"[FaceDetector] Warning: Could not initialize MTCNN ({e}). Using fallback crop.")

    def center_crop(self, img: Image.Image) -> Image.Image:
        """Fallback: Square center crop of the image."""
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return img.crop((left, top, right, bottom))

    def detect_and_crop(self, img: Image.Image) -> tuple[Image.Image, bool, list | None]:
        """
        Detect face in PIL Image and crop with margin.

        Returns:
            cropped_img (PIL.Image): Cropped face or center cropped image.
            face_detected (bool): True if a face was detected by MTCNN, False otherwise.
            bbox (list | None): Bounding box [x1, y1, x2, y2] if detected.
        """
        if img is None:
            raise ValueError("Input image cannot be None")

        img = img.convert("RGB")
        w, h = img.size

        if self.mtcnn is not None:
            try:
                boxes, probs = self.mtcnn.detect(img)
                if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.8:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]

                    # Add margin around face
                    box_w = x2 - x1
                    box_h = y2 - y1
                    margin_x = int(box_w * (self.margin / 100.0))
                    margin_y = int(box_h * (self.margin / 100.0))

                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)

                    cropped = img.crop((x1, y1, x2, y2))
                    return cropped, True, [x1, y1, x2, y2]
            except Exception as e:
                pass  # Fall back to center crop on detection error

        # Fallback if no face detected or MTCNN unavailable
        return self.center_crop(img), False, None
