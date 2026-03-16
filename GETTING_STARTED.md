# Getting Started in 5 Minutes

## 1. Clone & Install

```bash
git clone <repo-url>
cd modelCreation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Launch the App

```bash
python -m frontend.app
```

The app creates the data directory automatically. You'll see the main window with an empty image list.

## 3. Add Images

### Option A: Direct image files
1. Drop PNG/JPG files into `ml/data/raw_images/`
2. Click **Refresh** in the app

### Option B: PDF files
1. Drop PDF files into `ml/data/raw_pdfs/`
2. Click **Rescan PDFs** — converts to PNG automatically

## 4. Create Classes

1. In the **Classes** panel, click **Add**
2. Enter class name (e.g., "cat")
3. Pick a color
4. Repeat for each class

## 5. Annotate

1. Select a class (or press `1`–`9`)
2. Click and drag on the image to draw a box
3. Press `A`/`D` to navigate images
4. Press `S` to undo

That's it! The annotation is saved automatically.

## 6. Train

Once you have at least 30–50 annotations (more is better):

1. Click **Train** → set split percentages → confirm
2. App exports COCO JSONs automatically
3. Click **Train Model**
4. Watch the progress dialog
5. Training is very fast on small datasets (~20–30 seconds for ~100 images)

## 7. Predict

1. Select an image
2. Click **Test**
3. Red boxes appear with confidence scores
4. Review and click **Burn Predictions** to accept them

Then go back to Step 6 to retrain with the new data.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` / `D` | Previous / Next image |
| `1`–`9` | Select class |
| `S` | Undo last box |
| `W` | Delete mode (toggle) |
| Scroll | Zoom |

---

## Troubleshooting

**ModuleNotFoundError: No module named 'PyQt6'**
→ Run `pip install -r requirements.txt`

**CUDA out of memory**
→ Reduce batch size in training dialog (try 32 or 48)

**App is slow**
→ GPU? Check `import torch; print(torch.cuda.is_available())`

**No images showing up**
→ Make sure images are in `ml/data/raw_images/`. Click **Refresh**.

---

For more details, see the full [README.md](README.md).
