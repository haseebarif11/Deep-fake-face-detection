import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm.auto import tqdm


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    """
    Trains model for one epoch using Automatic Mixed Precision (AMP) if enabled.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training Batch", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device, use_amp=True):
    """
    Validates model performance for one epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation Batch", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def evaluate_classifier(model, loader, device, class_names=None, dataset_name="Test Set", use_amp=True):
    """
    Full evaluation routine computing Accuracy, Macro Precision, Recall, F1, ROC-AUC,
    and returning detailed evaluation metrics dictionary.
    """
    if class_names is None:
        class_names = ["fake", "real"]

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating [{dataset_name}]"):
            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Fake ROC-AUC (label 0 = fake, target binary indicator = (labels == 0))
    from sklearn.metrics import roc_auc_score
    fake_true = (all_labels == 0).astype(int)
    fake_probs = all_probs[:, 0]
    try:
        auc_fake = roc_auc_score(fake_true, fake_probs)
    except Exception:
        auc_fake = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)

    metrics = {
        "dataset_name": dataset_name,
        "total_samples": len(all_labels),
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_fake": float(auc_fake),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    print(f"\n==================================================")
    print(f" Evaluation Results: {dataset_name}")
    print(f"==================================================")
    print(f" Total Samples:     {len(all_labels)}")
    print(f" Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f" Macro Precision:   {precision_macro:.4f}")
    print(f" Macro Recall:      {recall_macro:.4f}")
    print(f" Macro F1 Score:    {f1_macro:.4f}")
    print(f" Fake ROC-AUC:      {auc_fake:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"==================================================\n")

    return metrics, all_labels, all_preds, all_probs
