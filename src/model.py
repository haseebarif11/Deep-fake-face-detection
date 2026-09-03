import os
import shutil
import tempfile
import torch
import torch.nn as nn
import torchvision.models as models


def build_efficientnet_b3(num_classes=2, dropout=0.4, pretrained=True):
    """
    Constructs EfficientNet-B3 model architecture for binary DeepFake detection.
    Class order: 0 = fake / AI-generated, 1 = real / authentic.
    Default classifier matches saved checkpoint architecture:
    Dropout(0.4) -> Linear(1536, 512) -> SiLU() -> Dropout(0.4) -> Linear(512, 2)
    """
    if pretrained:
        try:
            weights = models.EfficientNet_B3_Weights.DEFAULT
            model = models.efficientnet_b3(weights=weights)
        except Exception:
            model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    else:
        model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(512, num_classes)
    )
    return model


def load_model_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Loads saved model weights from a .pth file, PyTorch model checkpoint, or unzipped directory.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights path does not exist: {weights_path}")

    loaded_obj = None

    # Handle directory format (e.g. unzipped PyTorch archive folder)
    if os.path.isdir(weights_path):
        import zipfile
        temp_pth = os.path.join(tempfile.gettempdir(), "temp_model_archive.pth")
        try:
            with zipfile.ZipFile(temp_pth, 'w', compression=zipfile.ZIP_STORED) as z:
                for root, _, files in os.walk(weights_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, weights_path)
                        arc_name = os.path.join("archive", rel_path).replace("\\", "/")
                        z.write(full_path, arcname=arc_name)

            try:
                loaded_obj = torch.load(temp_pth, map_location=device, weights_only=True)
            except Exception:
                loaded_obj = torch.load(temp_pth, map_location=device, weights_only=False)
        finally:
            if os.path.exists(temp_pth):
                os.remove(temp_pth)
    else:
        # File path
        try:
            loaded_obj = torch.load(weights_path, map_location=device, weights_only=True)
        except Exception:
            loaded_obj = torch.load(weights_path, map_location=device, weights_only=False)

    state_dict = loaded_obj
    if isinstance(loaded_obj, nn.Module):
        model = loaded_obj
        model = model.to(device)
        model.eval()
        return model
    elif isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj:
            state_dict = loaded_obj["state_dict"]
        elif "model_state_dict" in loaded_obj:
            state_dict = loaded_obj["model_state_dict"]

    # Adapt classifier if state_dict has single-layer classifier (classifier.1.weight shape [2, 1536])
    if "classifier.1.weight" in state_dict and state_dict["classifier.1.weight"].shape == torch.Size([2, 1536]):
        in_features = model.classifier[1].in_features if hasattr(model.classifier[1], 'in_features') else 1536
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 2)
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
