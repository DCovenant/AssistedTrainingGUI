# Assisted Model Creation System

An interactive desktop app for annotating images and training CLIP-based object detection models with active learning. Label once, train, predict, validate predictions, and retrain—in a loop.

**Example:** Tested on 93 cat images with manual bounding box annotations.

---

## What This Does

1. **Import images** — Drop PDFs or image folders
2. **Annotate manually** — Draw boxes, assign classes via GUI
3. **Train a model** — Fine-tune CLIP on your annotations
4. **Run predictions** — Sliding-window inference on new images
5. **Review & burn** — Accept predictions as new training data
6. **Retrain** — Loop back to step 3

```
Images → Annotate → Train → Predict → Review → Burn → Retrain
```

No external APIs. Works offline. GPU-accelerated training (3–6 hours on RTX 5070).

---

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA (highly recommended for training; CPU works but slow)
- ~12GB VRAM for batch size 96

### Install

```bash
git clone <repo-url>
cd modelCreation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install PyQt6 torch torchvision transformers pymupdf Pillow matplotlib numpy

# For CUDA support, install PyTorch with your CUDA version:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Launch

```bash
python -m frontend.app
```

On first launch, the app will create the data directory structure:
```
ml/
├── data/
│   ├── raw_pdfs/          # Drop PDFs here
│   ├── raw_images/        # Converted images (auto-generated)
│   ├── coco/              # COCO JSON exports
│   └── data.db            # Annotation database
└── models/
    └── versions/          # Checkpoints (best_model, final_model)
```

---

## Workflow

### Step 1: Import Images

Place image files (PNG, JPG) in `ml/data/raw_images/` or PDF files in `ml/data/raw_pdfs/`.

If using PDFs:
- App converts to PNG at 2× resolution automatically
- Click **Rescan PDFs** to process new files

### Step 2: Define Classes

1. Click **Add** in the Classes panel
2. Enter class name (e.g., "cat", "dog")
3. Pick a color for visual distinction
4. Repeat for each class

### Step 3: Annotate Images

1. Select a class from the list (or press `1`–`9` for quick selection)
2. Click and drag on the image to draw a bounding box
3. Navigate with **A** (previous) / **D** (next)
4. Undo with **S**, delete annotations with **W** (delete mode)

The annotation is automatically saved to SQLite.

### Step 4: Split & Export

1. Click **Train** to open the split dialog
2. Set train/dev/test percentages (default: 70/15/15)
3. Confirm to split
   - Only annotated images are divided
   - Unlabeled images stay available for future labeling
4. App exports COCO JSON files automatically

### Step 5: Train the Model

1. Click **Train Model** to open training dialog
2. Adjust hyperparameters if desired:
   - Epochs: 20 (default)
   - Batch size: 96 (for RTX 5070; reduce for smaller GPUs)
   - Learning rate: 1e-5 (fine-tuning rate)
3. Click **Start Training**
4. Watch real-time loss/accuracy graphs
5. Click **Stop** to end early

Training saves:
- **best_model/** — Best by validation accuracy
- **final_model/** — Final epoch checkpoint

### Step 6: Run Predictions

1. Select an image
2. Click **Test** to run inference
3. Red boxes appear with class name and confidence score
4. Review predictions:
   - **Burn Predictions** — Accept them as annotations (adds to training set)
   - **Undo Predictions** — Clear them without saving

### Step 7: Retrain (Active Learning Loop)

Repeat steps 4–6 to incorporate predictions into the training set. With each iteration:
- More labeled data
- Better model accuracy
- Fewer manual annotations needed

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` | Previous image |
| `D` | Next image |
| `S` | Undo last annotation |
| `W` | Toggle delete mode |
| `X` | Remove image from dataset |
| `Z` | Undo image removal |
| `1`–`9` | Select class by position |
| Mouse wheel | Zoom in/out |
| Click + drag | Pan (no class) / Draw box (class selected) |

---

## Example: Cat Detection

### Data Used

- **93 cat images** from [Microsoft Cats vs Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- **93 cat bounding box annotations** (one per image)
- **Train/Dev/Test split:** 42 / 19 / 32 images

### To Reproduce

1. Download cat images from the dataset
2. Place in `ml/data/raw_images/`
3. Annotate with class "cat"
4. Train as described in the workflow section

### Notes

This was a small-scale test. With more images and annotations, accuracy would improve. The system works well with limited labeled data due to CLIP's pre-training.

---

## Data Format

### Annotation Database (SQLite)

Stored in `ml/data/data.db`:

| Table | Columns |
|-------|---------|
| `classes` | id, name, color |
| `annotations` | id, image_id, class_id, x, y, width, height, text |
| `image_splits` | image_id, split (train/dev/test) |

### COCO JSON Export

Generated in `ml/data/coco/`:
- `coco_train.json` — Training set
- `coco_dev.json` — Validation set
- `coco_test.json` — Test set (if created)

Format matches [COCO object detection spec](https://cocodataset.org/#format-data).

---

## How It Works (Technical)

### Model Architecture

```
Image crop → CLIP ViT-B/32 encoder → 512-dim features → Classification head → Class prediction
```

- **Base:** OpenAI's CLIP (Vision Transformer)
- **Frozen:** Most of CLIP's layers (early feature extraction)
- **Fine-tuned:** Last 4 transformer layers + 2-layer classification head
- **Training:** AdamW optimizer, cosine annealing, FP16 mixed precision
- **Loss:** Weighted cross-entropy (handles class imbalance)

### Inference

1. Slide multiple window sizes across image (3 scales: 190×180, 250×230, 320×300)
2. Classify each crop with trained model
3. Keep high-confidence predictions (threshold: 0.6)
4. Remove overlapping boxes with GPU-accelerated NMS

### Active Learning

- Predictions are shown as separate "prediction" boxes
- User reviews and decides to accept (burn) or reject
- Accepted predictions are converted to annotations
- Next training run includes these burned predictions as ground truth

---

## Performance

### Training

| Metric | Value |
|--------|-------|
| Data | ~42 training images, 19 dev, 32 test |
| Annotations | 93 total (cat bounding boxes) |
| Augmentation | 16× (3 rotations × 2 color modes) |
| Batch size | 96 |
| Time (20 epochs) | 20–30 seconds on RTX 5070 |
| Training scale | Very fast on small datasets (~100 images) |

### Inference

| Metric | Value |
|--------|-------|
| Per-image latency | 2–5 seconds (RTX 5070) |
| Bottleneck | Sliding window inference, not training |
| GPU memory | ~10GB peak |

---

## Troubleshooting

### GPU Memory Error

If you get CUDA out-of-memory errors:
1. Reduce batch size (e.g., 96 → 48 or 32)
2. Reduce dataloader workers (4 → 2)
3. Use CPU (slow, but works)

**In `backend/fine_tuning/training.py`:**
```python
train_loader = DataLoader(
    train_dataset, batch_size=48,  # Reduce here
    shuffle=True,
    pin_memory=True, num_workers=2,  # Reduce workers
    persistent_workers=True
)
```

### Training on CPU

Training without a GPU is possible but significantly slower. On small datasets (~100 images), GPU training takes 20–30 seconds. CPU training would be substantially longer.

If you must train on CPU:
1. Reduce batch size (32 or 16)
2. Reduce number of workers
3. Be patient—it will take minutes to hours depending on dataset size

### No GPU Detected

Check PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Validation & Debugging

### Check COCO JSON Integrity

```bash
python -m backend.data.validate_coco
```

### Check Images for Corruption

```bash
python -m backend.data.validate_images
```

### Profile DataLoader Performance

```bash
python -m backend.fine_tuning.profile_dataloader
```

---

## Project Structure

```
modelCreation/
├── frontend/                        # PyQt6 GUI
│   ├── app.py                       # Main application
│   └── widgets/
│       ├── image_viewer.py          # Zoomable image canvas
│       ├── image_selector.py        # Image browser
│       ├── class_config_dialog.py   # Class creation
│       ├── dataset_division_dialog.py
│       ├── background_preview.py    # Visualize background crops
│       └── training_progress_dialog.py
│
├── backend/
│   ├── data/
│   │   ├── annotation_database.py   # SQLite ORM
│   │   ├── pdf_converter.py         # PDF → PNG
│   │   ├── coco_exporter.py         # COCO JSON export
│   │   ├── dataset_splitter.py      # Train/dev/test split
│   │   ├── validate_coco.py         # JSON validation
│   │   └── validate_images.py       # Image corruption check
│   │
│   └── fine_tuning/
│       ├── training.py              # Training loop
│       ├── inference.py             # Sliding window + NMS
│       └── background_generator.py  # Negative sample generation
│
├── ml/
│   ├── data/                        # Dataset directory (auto-created)
│   ├── models/                      # Checkpoints (auto-created)
│   └── configs/
│
└── README.md                        # This file
```

---

## License

MIT License. Free to use, modify, and distribute.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

See `LICENSE` file for full text.

---

## Contributing

Found a bug? Have an improvement idea? Feel free to open an issue or PR.

---

## Citation

If you use this in research or projects, please cite:

```bibtex
@software{assisted_model_creation,
  title = {Assisted Model Creation System},
  author = {Author Name},
  year = {2024},
  url = {https://github.com/your-username/modelCreation}
}
```

---

## FAQ

**Q: Do I need GPU?**
A: Highly recommended for training. CPU works but is 10–20× slower.

**Q: How much data do I need?**
A: Minimum ~50 annotated regions to start training. More data = better accuracy. The system was tested with 93 cat images.

**Q: Can I add my own model?**
A: Yes. Modify `CLIPDetector` in `backend/fine_tuning/training.py` to use a different backbone.

**Q: Can I export predictions to COCO format?**
A: Yes. Burned predictions are automatically added to the SQLite database and exported as COCO JSON.

**Q: Does it work on Mac?**
A: Yes, on Mac with Apple Silicon (GPU acceleration available). CPU-only slower.

---

## Contact

Questions or feedback? Open an issue on GitHub.

Happy annotating! 🎯
