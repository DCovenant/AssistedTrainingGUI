# BT-7274 — Electrical Schematic Object Detection

A desktop application for annotating electrical schematic diagrams (EPLAN PDFs) and training a CLIP-based object detection model to automatically detect components like terminals, junctions, and other elements.

Built with PyQt6 for the annotation interface and fine-tuned OpenAI CLIP (ViT-B/32) for detection.

---

## How It Works

The application follows an **active learning loop**:

1. **Import** — Drop EPLAN-exported PDFs into the data folder; the app converts them to high-resolution PNGs
2. **Annotate** — Draw bounding boxes on schematic images and assign classes (terminal, junction, etc.)
3. **Split** — Divide annotated images into train/dev/test sets (unlabeled images are reserved for future labeling)
4. **Export** — Generate COCO-format JSON files from annotations
5. **Train** — Fine-tune CLIP's vision encoder to classify cropped regions
6. **Predict** — Run sliding-window inference on new images
7. **Burn** — Accept model predictions as annotations, growing the training set

```
PDF Schematics → PNG Images → Annotate → COCO JSON → Train CLIP → Predict → Review/Burn → Repeat
```

---

## Architecture

### Model — CLIPDetector

```
Image Crop → CLIP ViT-B/32 Vision Encoder → 512-dim features → Classification Head → Class Prediction
                (last 2 layers unfrozen)                         (256 → num_classes)
```

- **Base model**: `openai/clip-vit-base-patch32`
- **Training strategy**: Freeze most of the vision encoder, fine-tune only the last 2 transformer layers + a custom classification head
- **Background class**: Automatically generated from unselected image regions using a systematic grid, added as class 0
- **Inference**: Multi-scale sliding window scans the full image, CLIP classifies each crop, Non-Max Suppression removes overlaps
- **Mixed precision** (FP16) training with cosine annealing LR schedule

### Data Pipeline

```
raw_pdfs/           →  PDF-to-PNG conversion (PyMuPDF, 2x resolution)
raw_images/         →  Annotation via GUI → SQLite database (data.db)
data.db             →  Dataset splitter (train/dev/test) → COCO JSON export
coco/*.json         →  Training input (COCOCropDataset)
models/versions/    →  Saved checkpoints (best_model, final_model)
```

### Database Schema (SQLite)

| Table | Purpose |
|---|---|
| `classes` | Annotation class definitions (name, color) |
| `annotations` | Bounding boxes linked to images and classes (x, y, width, height, text) |
| `image_splits` | Train/dev/test assignment per image |

---

## Project Structure

```
AnomalyDetectionML/
├── frontend/                      # PyQt6 GUI
│   ├── app.py                     # Main application window (TerminalDetectorApp)
│   └── widgets/
│       ├── image_viewer.py        # Zoomable image viewer with annotation overlay
│       ├── image_selector.py      # Image selection/filtering tool
│       ├── class_config_dialog.py # Add new annotation class (name + color)
│       ├── class_removal_dialog.py
│       ├── annotation_text_dialog.py
│       ├── dataset_division_dialog.py  # Train/dev/test split percentages
│       ├── background_preview.py       # Visualize annotated vs background regions
│       └── training_progress_dialog.py # Real-time training with loss/accuracy graph
│
├── backend/
│   ├── data/
│   │   ├── annotation_database.py # SQLite database for annotations and classes
│   │   ├── pdf_converter.py       # PDF → PNG conversion (PyMuPDF)
│   │   ├── data_checker.py        # Check availability of PDFs/images
│   │   ├── coco_exporter.py       # Export annotations to COCO JSON format
│   │   ├── dataset_splitter.py    # Train/dev/test split with active learning support
│   │   ├── image_metadata.py      # Image dimension extraction
│   │   ├── validate_coco.py       # Validate COCO JSON integrity
│   │   ├── validate_images.py     # Check images for corruption
│   │   └── run_coco_export.py     # CLI script to export COCO JSONs
│   │
│   └── fine_tuning/
│       ├── training.py            # CLIP fine-tuning (COCOCropDataset, CLIPDetector, TrainingLauncher)
│       ├── inference.py           # Sliding window detection + NMS
│       ├── background_generator.py # Generate negative samples from unselected regions
│       └── profile_dataloader.py  # DataLoader performance profiler
│
├── ml/
│   ├── data/
│   │   ├── raw_pdfs/              # Input: EPLAN PDF schematics
│   │   ├── raw_images/            # Converted PNG images
│   │   ├── coco/                  # Exported COCO JSON files (train/dev/test)
│   │   └── data.db                # SQLite annotation database
│   ├── models/
│   │   └── versions/
│   │       ├── best_model/        # Best checkpoint by dev accuracy
│   │       └── final_model/       # Final epoch checkpoint
│   └── configs/
│       └── split_configs.json
│
└── extract_eplan_categories.py    # Utility: extract component categories from PDF text
```

---

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA (recommended for training)

### Dependencies

| Package | Purpose |
|---|---|
| `PyQt6` | Desktop GUI framework |
| `torch` | Deep learning framework |
| `transformers` | Hugging Face — CLIP model and processor |
| `PyMuPDF` (`fitz`) | PDF to image conversion |
| `Pillow` | Image processing |
| `matplotlib` | Training progress graphs |

### Install

```bash
pip install PyQt6 torch torchvision transformers pymupdf Pillow matplotlib
```

> For CUDA support, install PyTorch with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

---

## Usage

### Launch the Application

```bash
cd AnomalyDetectionML
python -m frontend.app
```

On first launch, if PDFs are found in `ml/data/raw_pdfs/` but no images exist yet, the app will prompt to convert them.

### Workflow

#### 1. Import PDFs

Place EPLAN-exported PDF schematics in `ml/data/raw_pdfs/`. Use the **Rescan PDFs** button to convert new PDFs to PNG at any time.

#### 2. Define Classes

Click **Add** in the Classes panel to create annotation classes (e.g., "Terminal", "Junction"). Each class gets a name and a color for visual distinction.

#### 3. Annotate Images

1. Select a class from the panel (or press `1`-`9`)
2. Click and drag on the image to draw a bounding box
3. Navigate images with `A`/`D` keys
4. Undo with `S`, toggle delete mode with `W`

#### 4. Split and Export

1. Click **Train** to open the dataset split dialog
2. Set train/dev/test percentages (default: 70/15/15)
3. Confirm to split — only annotated images are divided; unannotated images remain available for future labeling
4. The app will prompt to export COCO JSON files

#### 5. Train the Model

Click **Train Model** to open the training dialog. Training runs in a background thread with:
- Live batch progress bar
- Real-time loss and accuracy plots (train + dev)
- Elapsed time and ETA
- Stop button for early termination

The best model (by dev accuracy) is automatically saved to `ml/models/versions/best_model/`.

#### 6. Run Inference

1. Select an image
2. Click **Test** to run the trained model
3. Predictions appear as red bounding boxes with class name and confidence
4. Click **Burn Predictions** to save them as annotations (active learning)
5. Click **Undo Predictions** to clear them

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `A` | Previous image |
| `D` | Next image |
| `S` | Undo last annotation |
| `W` | Toggle delete mode |
| `X` | Remove image from dataset |
| `Z` | Undo image removal |
| `1`-`9` | Select class by position |
| Mouse wheel | Zoom in/out |
| Click + drag | Pan (no class selected) / Annotate (class selected) |

---

## Training Details

- **Optimizer**: AdamW (lr=1e-5)
- **Scheduler**: Cosine annealing
- **Batch size**: 96 (tuned for RTX 5070)
- **Epochs**: 20 (default)
- **Mixed precision**: FP16 via `torch.amp`
- **Background generation**: Systematic grid with IoU < 0.1 threshold against annotations
- **Sliding window inference**: 6 window sizes (80x50 to 150x100), 50% stride overlap, confidence threshold 0.7, NMS IoU threshold 0.3

---

## Validation Utilities

```bash
# Validate COCO JSON files for invalid bounding boxes
python -m backend.data.validate_coco

# Validate images for corruption and format issues
python -m backend.data.validate_images

# Export COCO JSONs from command line
python -m backend.data.run_coco_export

# Profile DataLoader performance
python -m backend.fine_tuning.profile_dataloader
```

---

## License

Private project — all rights reserved.
