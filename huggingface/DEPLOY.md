# 🎙️ Stutter Analysis - Hugging Face Deployment Guide

## Quick Deploy to Hugging Face Spaces

### Step 1: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in:
   - **Space name:** `stutter-analysis`
   - **License:** MIT
   - **SDK:** Gradio
   - **Hardware:** CPU Basic (free) or GPU if available
4. Click **"Create Space"**

### Step 2: Upload Files

Upload these files to your Space:

```
huggingface/
├── app.py                              # Main Gradio app
├── requirements.txt                    # Dependencies
└── wavlm_stutter_classification_best.pth  # Your trained model
```

**To upload the model checkpoint:**
```bash
# Option 1: Git LFS (for large files)
cd your-space-repo
git lfs install
git lfs track "*.pth"
cp ../checkpoints/wavlm_stutter_classification_best.pth .
git add .
git commit -m "Add model checkpoint"
git push

# Option 2: Hugging Face Hub UI
# Go to your Space > Files > Upload files
# Upload wavlm_stutter_classification_best.pth
```

### Step 3: Configure Space Settings

In your Space settings:
- **Python version:** 3.9+
- **Startup duration:** Extended (models take time to load)

### Step 4: Access Your App

Once deployed, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/stutter-analysis
```

---

## Alternative: Run Locally with Gradio

```bash
cd huggingface
pip install -r requirements.txt

# Copy your model checkpoint
cp ../checkpoints/wavlm_stutter_classification_best.pth .

# Run
python app.py
```

Opens at: http://localhost:7860

---

## Troubleshooting

### "Out of Memory" Error
- Use CPU instead of GPU
- Or upgrade to paid GPU tier

### "Model not found" Error
- Ensure `wavlm_stutter_classification_best.pth` is in the same folder as `app.py`

### Slow First Load
- Normal! Models download on first run (~1-2 min)
- Subsequent loads are faster

---

## Share Your Space

Once deployed, share the link:
```
https://huggingface.co/spaces/YOUR_USERNAME/stutter-analysis
```

Anyone can use it without installing anything! 🎉
