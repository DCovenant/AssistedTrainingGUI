# Assisted Model Creation System

An interactive desktop app for annotating images and training a CLIP-based object detection model with active learning. Label, train, predict, review, and retrain—in a loop.

**Tested on:** 93 cat images from [Microsoft Cats vs Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). Training took ~20–30 seconds on an RTX 5070.

---

## What This Does

1. **Import images** — Drop PDFs or image files into the data folder
2. **Annotate manually** — Draw bounding boxes and assign classes via GUI
3. **Train a model** — Fine-tune CLIP on your annotations
4. **Run predictions** — Sliding-window inference on new images
5. **Review & burn** — Accept good predictions as new training data
6. **Retrain** — Loop back to step 3

```
Images → Annotate → Train → Predict → Review → Burn → Retrain
```

No external APIs. Works offline.

---

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA (developed and tested on RTX 5070)

> **Note:** The code has a CPU fallback (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`), but training and inference have only been tested on NVIDIA GPUs. CPU and Mac are untested.

### Install

```bash
git clone <repo-url>
cd modelCreation

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

# For CUDA support, install PyTorch with your CUDA version:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Launch

```bash
python -m frontend.app
```

On first launch, the app creates the data directory structure:
```
ml/
├── data/
│   ├── raw_pdfs/          # Drop PDFs here
│   ├── raw_images/        # Converted/imported images
│   ├── coco/              # COCO JSON exports (auto-generated)
│   └── data.db            # Annotation database (auto-created)
└── models/
    └── versions/          # Checkpoints (best_model, final_model)
```

---

## Workflow

### Step 1: Import Images

Place image files (PNG, JPG) in `ml/data/raw_images/` or PDF files in `ml/data/raw_pdfs/`.

If using PDFs:
- App converts to PNG at 2x resolution automatically
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
4. Undo with **S**, toggle delete mode with **W**

Annotations are saved to SQLite automatically.

### Step 4: Split & Export

1. Click **Train** to open the split dialog
2. Set train/dev/test percentages (default: 70/15/15)
3. Confirm — only annotated images are divided; unlabeled images stay available
4. App exports COCO JSON files automatically

### Step 5: Train the Model

1. Click **Train Model** to open training dialog
2. Click **Start Training**
3. Watch real-time loss/accuracy graphs
4. Click **Stop** to end early if needed

Training saves:
- **best_model/** — Best checkpoint by validation accuracy
- **final_model/** — Final epoch checkpoint

### Step 6: Run Predictions

1. Select an image
2. Click **Test** to run inference
3. Red boxes appear with class name and confidence score
4. Review predictions:
   - **Burn Predictions** — Accept them as annotations (adds to training set)
   - **Undo Predictions** — Clear them

### Step 7: Retrain (Active Learning)

Repeat steps 4–6 to grow the training set with accepted predictions.

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
| Click + drag | Pan (no class selected) / Draw box (class selected) |

---

## Example: Cat Detection

### Data Used

- **93 cat images** from the Microsoft Cats vs Dogs dataset
- **93 bounding box annotations** (one per image)
- **Split:** 42 train / 19 dev / 32 test
- **Training time:** ~20–30 seconds on RTX 5070

### To Reproduce

1. Download cat images from the dataset
2. Place in `ml/data/raw_images/`
3. Create class "cat" and annotate images
4. Train as described above

### Notes

This was a small-scale test. With only 42 training images, overfitting is expected (train accuracy reaches ~100% while dev accuracy lags behind). More annotations would improve generalization.

---

## How It Works (Technical)

### Model Architecture

```
Image crop → CLIP ViT-B/32 encoder → 512-dim features → Classification head → Class prediction
```

- **Base:** OpenAI CLIP (`openai/clip-vit-base-patch32`) — hardcoded
- **Frozen:** All layers except the last 4 transformer blocks
- **Fine-tuned:** Last 4 vision encoder layers + 2-layer classification head (Linear → ReLU → Dropout → Linear)
- **Training:** AdamW optimizer, cosine annealing LR, FP16 mixed precision, weighted cross-entropy loss
- **Early stopping:** Patience of 3 epochs on dev accuracy

### Inference

1. Slide 3 window sizes across the image (190x180, 250x230, 320x300)
2. Classify each crop with the trained model
3. Keep predictions above confidence threshold (0.6)
4. Remove overlapping boxes with GPU-accelerated NMS (torchvision)

### Active Learning

- Predictions are displayed as separate "prediction" boxes (red)
- User reviews and accepts (burns) or rejects them
- Accepted predictions become annotations in the database
- Next training run includes burned predictions as ground truth

---

## Performance (Tested)

### Training (93 images, RTX 5070)

| Metric | Value |
|--------|-------|
| Training images | 42 |
| Dev images | 19 |
| Test images | 32 |
| Annotations | 93 total |
| Augmentation | 6x (3 rotations x 2 color modes) |
| Batch size | 96 |
| Time (20 epochs) | ~20–30 seconds |

### Inference

| Metric | Value |
|--------|-------|
| Per-image | 2–5 seconds (depends on image size) |
| Bottleneck | Sliding window crop classification |

Training time scales with dataset size. Larger datasets will take longer.

---

## Known Limitations

- **NVIDIA GPU only (tested):** The code falls back to CPU if no GPU is found, but CPU training and inference have not been tested. It will be significantly slower.
- **Mac untested:** No Mac-specific code exists (no MPS backend support). It may work on CPU but this is unverified.
- **Model is hardcoded:** The CLIP model ID (`openai/clip-vit-base-patch32`) is hardcoded in 3 files (`training.py`, `inference.py`, `profile_dataloader.py`). Swapping to a different model requires changing all 3 and testing compatibility.
- **Checkpoints don't save model ID:** If you change the model and try to load an old checkpoint, it will fail silently or crash.
- **Hyperparameters are hardcoded:** Batch size, learning rate, window sizes, etc. are set in code, not config files.
- **No automated tests:** The data pipeline and training loop have no unit tests.
- **Logging is print-only:** No structured logging. Uses `print()` statements.
- **Small dataset overfitting:** With ~40 training images, the model memorizes quickly. Dev accuracy will lag behind train accuracy.

---

## Changing the Model

To use a different CLIP variant (e.g., `openai/clip-vit-large-patch14`), you need to update **3 files**:

1. `backend/fine_tuning/training.py` — `MODEL_ID` on line 251
2. `backend/fine_tuning/inference.py` — `MODEL_ID` on line 21
3. `backend/fine_tuning/profile_dataloader.py` — hardcoded string on line 14

The classification head reads `hidden_size` from the model config dynamically, so different embedding dimensions should work. But this is untested — verify before relying on it.

Only CLIP-based models from Hugging Face are supported. Using a non-CLIP model would require rewriting `CLIPDetector` and the data pipeline.

---

## Troubleshooting

### GPU Memory Error

Reduce batch size in the training dialog, or edit `backend/fine_tuning/training.py`:
```python
train_loader = DataLoader(
    train_dataset, batch_size=48,  # Reduce from 96
    shuffle=True,
    pin_memory=True, num_workers=2,  # Reduce from 4
    persistent_workers=True
)
```

### No GPU Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Validation Utilities

```bash
# Validate COCO JSON files
python -m backend.data.validate_coco

# Check images for corruption
python -m backend.data.validate_images

# Export COCO JSONs from CLI
python -m backend.data.run_coco_export

# Profile DataLoader performance
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
│   │   ├── annotation_database.py   # SQLite database
│   │   ├── pdf_converter.py         # PDF → PNG
│   │   ├── coco_exporter.py         # COCO JSON export
│   │   ├── dataset_splitter.py      # Train/dev/test split
│   │   ├── validate_coco.py         # JSON validation
│   │   └── validate_images.py       # Image corruption check
│   │
│   └── fine_tuning/
│       ├── training.py              # Training loop (CLIPDetector, TrainingLauncher)
│       ├── inference.py             # Sliding window + NMS
│       └── background_generator.py  # Negative sample generation
│
├── ml/
│   ├── data/                        # Dataset directory (auto-created)
│   ├── models/                      # Checkpoints (auto-created)
│   └── configs/
│
└── README.md
```

---

## Data Format

### SQLite Database (`ml/data/data.db`)

| Table | Columns |
|-------|---------|
| `classes` | id, name, color |
| `annotations` | id, image_id, class_id, x, y, width, height, text |
| `image_splits` | image_id, split (train/dev/test) |

### COCO JSON (`ml/data/coco/`)

- `coco_train.json`, `coco_dev.json`, `coco_test.json`
- Standard [COCO object detection format](https://cocodataset.org/#format-data)

---

## License

MIT License — free to use, modify, and distribute. See `LICENSE` file.

---

## Contributing

Found a bug or have an idea? Open an issue or PR.
