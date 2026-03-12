"""
Training script for CLIP-based object detection model.
Fine-tunes CLIP's vision encoder to classify cropped regions
(terminals, junctions, etc.) from schematic images.

How it works:
1. Load COCO annotations (bounding boxes + classes)
2. Crop each annotated region from the image
3. Generate background crops from all unselected regions
4. Fine-tune CLIP to classify these crops
5. At inference: sliding window finds candidates, CLIP classifies them
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW

from .background_generator import generate_background_crops

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# PART 1: DATASET - Crops annotated regions from images
# ============================================================================

class COCOCropDataset(Dataset):
    """
    Loads COCO annotations and crops each bounding box region from images.
    Generates background crops from all unselected regions systematically.

    Categories are shifted by +1 to make room for background at index 0:
      0 = background (systematic grid coverage)
      1 = terminal (or whatever your first class is)
      2 = junction, etc.
    """

    def __init__(self, coco_json_path: str, processor: CLIPProcessor,
                 images_directory: str | None = None):
        if images_directory is None:
            project_root = Path(__file__).parent.parent.parent
            images_directory = str(project_root / "ml" / "data" / "raw_images")

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        self.images_dir = Path(images_directory)
        self.processor = processor
        self._image_cache: dict[str, Image.Image] = {}

        # Build image lookup: id -> metadata
        self.id_to_image = {img['id']: img for img in coco_data['images']}

        # Categories: background at index 0, then original classes shifted by 1
        self.categories = [{'id': 0, 'name': 'background'}] + coco_data['categories']
        self.cat_id_to_idx = {cat['id']: idx + 1 for idx, cat in enumerate(coco_data['categories'])}
        self.num_classes = len(self.categories)

        # Positive samples: annotated regions
        self.samples = []
        for ann in coco_data['annotations']:
            img_meta = self.id_to_image[ann['image_id']]
            self.samples.append({
                'file_name': img_meta['file_name'],
                'bbox': ann['bbox'],
                'label': self.cat_id_to_idx[ann['category_id']]
            })

        num_positives = len(self.samples)

        # Background samples: systematic grid covering all unselected regions
        bg_samples = self._generate_all_background_crops(coco_data['images'], coco_data['annotations'])
        self.samples.extend(bg_samples)

        print(f"Dataset loaded: {num_positives} positives + {len(bg_samples)} background "
              f"from {len(self.id_to_image)} images, {self.num_classes} classes")

    def _generate_all_background_crops(
        self, images: list[dict], annotations: list[dict]
    ) -> list[dict]:
        """Generate background crops from all unselected regions in each image.

        Uses systematic grid covering to ensure all unselected space is represented.

        Args:
            images: List of image metadata dicts
            annotations: List of annotation dicts from COCO

        Returns:
            List of background crop samples
        """
        image_annotations: dict[int, list] = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann['bbox'])

        bg_samples = []
        for img_meta in images:
            img_id = img_meta['id']
            img_w, img_h = img_meta['width'], img_meta['height']
            anns = image_annotations.get(img_id, [])

            bg_crops = generate_background_crops(img_w, img_h, anns)
            for crop in bg_crops:
                bg_samples.append({
                    'file_name': img_meta['file_name'],
                    'bbox': crop['bbox'],
                    'label': 0
                })

        return bg_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_name = sample['file_name']

        # Load image (cached)
        if file_name not in self._image_cache:
            image_path = self.images_dir / file_name
            self._image_cache[file_name] = Image.open(image_path).convert('RGB')
        image = self._image_cache[file_name]

        # Crop and process
        x, y, w, h = sample['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Validate crop bounds
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            # Return background crop as fallback
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

        x1, y1 = max(0, x), max(0, y)
        x2 = min(image.width, x + w)
        y2 = min(image.height, y + h)

        # Skip if crop is outside image bounds
        if x2 <= x1 or y2 <= y1:
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

        try:
            crop = image.crop((x1, y1, x2, y2)).convert('RGB')

            # Validate crop has content
            if crop.size[0] <= 0 or crop.size[1] <= 0:
                return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

            inputs = self.processor(images=crop, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            label = torch.tensor(sample['label'], dtype=torch.long)
            return pixel_values, label
        except Exception as e:
            # Fallback: return blank tensor on any processing error
            print(f"Warning: Failed to process crop at {file_name} bbox {sample['bbox']}: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(0, dtype=torch.long)


# ============================================================================
# PART 2: CLASSIFICATION HEAD - Sits on top of CLIP's vision encoder
# ============================================================================

class CLIPDetector(nn.Module):
    """
    CLIP vision encoder + classification head.

    Architecture:
        Image crop → CLIP vision encoder → 512-dim features → Classification head → class prediction

    We freeze most of CLIP and only train the classification head
    + last few layers of the vision encoder.
    """

    def __init__(self, clip_model: CLIPModel, num_classes: int):
        super().__init__()
        self.vision_model = clip_model.vision_model

        # Classification head: takes CLIP's 512-dim output → num_classes
        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Freeze most of vision encoder, only train last 2 layers
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.vision_model.encoder.layers[-2:].parameters():
            param.requires_grad = True

        # Classification head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, pixel_values):
        # Get CLIP vision features
        vision_output = self.vision_model(pixel_values=pixel_values)
        # Use the [CLS] token embedding (pooled output)
        features = vision_output.pooler_output
        # Classify
        logits = self.classifier(features)
        return logits


# ============================================================================
# PART 3: TRAINING LAUNCHER
# ============================================================================

class TrainingLauncher:
    """Main training orchestrator"""

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self):
        self.model = None
        self.processor = None

    def _load_model(self, num_classes: int):
        """Load CLIP model and add classification head."""
        if self.model is not None:
            return

        print("\nLoading CLIP model...")

        clip_model = CLIPModel.from_pretrained(self.MODEL_ID)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)

        # Create detector: CLIP encoder + classification head
        self.model = CLIPDetector(clip_model, num_classes).to(device)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {trainable:,}/{total:,} trainable parameters")

    def train(self,
              epochs: int = 20,
              batch_size: int = 96,
              learning_rate: float = 1e-5,
              save_dir: str | None = None,
              on_batch: Callable[[int, int], None] | None = None,
              on_epoch: Callable[[dict], None] | None = None,
              should_stop: Callable[[], bool] | None = None):
        """Full training loop with validation.

        Args:
            epochs: Number of passes through entire training data (default 5)
            batch_size: Images per batch (default 96 for RTX 5070)
            learning_rate: 1e-4 for faster convergence with frozen backbone
            save_dir: Where to save the trained model. If None, uses default ml/models/versions
            on_batch: Callback(current_batch, total_batches) for batch progress
            on_epoch: Callback(metrics_dict) after each epoch validation
            should_stop: Callback() that returns True if training should stop

        Raises:
            Exception: If training fails at any stage
        """
        if save_dir is None:
            project_root = Path(__file__).parent.parent.parent
            save_dir = str(project_root / "ml" / "models" / "versions")

        # Load datasets
        temp_processor = CLIPProcessor.from_pretrained(self.MODEL_ID)

        project_root = Path(__file__).parent.parent.parent
        train_json = str(project_root / "ml" / "data" / "coco" / "coco_train.json")
        dev_json = str(project_root / "ml" / "data" / "coco" / "coco_dev.json")
        images_dir = str(project_root / "ml" / "data" / "raw_images")

        train_dataset = COCOCropDataset(train_json, temp_processor, images_dir)
        dev_dataset = COCOCropDataset(dev_json, temp_processor, images_dir)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=4, persistent_workers=True
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=2, persistent_workers=True
        )

        num_classes = train_dataset.num_classes
        self._load_model(num_classes)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler('cuda')

        best_dev_acc = 0.0
        total_batches = len(train_loader) + len(dev_loader)

        for epoch in range(epochs):
            # Check if training should stop
            if should_stop and should_stop():
                print("Training stopped by user")
                break

            # --- TRAIN ---
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (pixel_values, labels) in enumerate(train_loader):
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with autocast('cuda'):
                    logits = self.model(pixel_values)
                    loss = criterion(logits, labels)

                train_loss += loss.item()
                predictions = logits.argmax(dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if on_batch:
                    on_batch(batch_idx + 1, total_batches)

            train_acc = train_correct / train_total * 100
            avg_train_loss = train_loss / len(train_loader)

            # --- VALIDATE ---
            self.model.eval()
            dev_loss = 0
            dev_correct = 0
            dev_total = 0

            with torch.no_grad():
                for batch_idx, (pixel_values, labels) in enumerate(dev_loader):
                    pixel_values = pixel_values.to(device)
                    labels = labels.to(device)

                    logits = self.model(pixel_values)
                    loss = criterion(logits, labels)
                    dev_loss += loss.item()

                    predictions = logits.argmax(dim=1)
                    dev_correct += (predictions == labels).sum().item()
                    dev_total += labels.size(0)

                    if on_batch:
                        on_batch(len(train_loader) + batch_idx + 1, total_batches)

            dev_acc = dev_correct / dev_total * 100
            avg_dev_loss = dev_loss / len(dev_loader)

            metrics = {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'dev_loss': avg_dev_loss,
                'dev_acc': dev_acc,
                'is_best': dev_acc > best_dev_acc
            }

            scheduler.step()

            if on_epoch:
                on_epoch(metrics)

            # Save best model
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                self._save_checkpoint(
                    Path(save_dir) / "best_model", num_classes,
                    train_dataset.categories, dev_acc, epoch + 1
                )

        # Final save
        self._save_checkpoint(
            Path(save_dir) / "final_model", num_classes,
            train_dataset.categories, dev_acc, epochs
        )

        return best_dev_acc

    def _save_checkpoint(self, path: Path, num_classes: int,
                         categories: list, dev_acc: float, epoch: int) -> None:
        """Save model checkpoint to disk.

        Args:
            path: Directory to save checkpoint
            num_classes: Number of output classes
            categories: List of category dicts
            dev_acc: Dev set accuracy at this checkpoint
            epoch: Epoch number
        """
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': num_classes,
            'categories': categories,
            'dev_accuracy': dev_acc,
            'epoch': epoch
        }, path / "model.pt")
