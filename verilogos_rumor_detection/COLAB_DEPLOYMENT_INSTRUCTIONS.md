# 🚀 Google Colab Deployment Instructions

## 📦 Package Contents

Your deployment package has been successfully created with the following components:

### Files Created:
1. **colab_deployment.zip** (12 MB)
   - Complete project source code
   - All configuration files
   - ACL2017 dataset (Twitter15 + Twitter16)
   - Requirements and dependencies
   - Total: 2,341 files

2. **Run_on_Colab.ipynb** (7 KB)
   - Ready-to-use Colab notebook
   - Automated pipeline orchestration
   - Results packaging and Drive backup

---

## 📋 What's Included in the Zip

### Core Components:
- ✅ `run.py` - Main CLI entrypoint
- ✅ `requirements.txt` - All Python dependencies
- ✅ `README.md` - Project documentation
- ✅ `configs/` - 4 configuration files:
  - `default.yaml`
  - `colab_tda_only.yml`
  - `colab_text_only.yaml`
  - `colab_hybrid.yaml`

### Source Code (`src/topolie/`):
- ✅ `data/loaders.py` - Dataset loading and preprocessing
- ✅ `tda/pipeline.py` - Topological Data Analysis pipeline
- ✅ `models/hybrid.py` - Hybrid neural network architecture
- ✅ `models/trainer.py` - Training loop and optimization
- ✅ `eval/visualizer.py` - Metrics visualization
- ✅ `experiments/runner.py` - Experiment orchestration

### Dataset (`data/acl2017/`):
- ✅ Twitter15: 742 rumor propagation events
- ✅ Twitter16: 592 rumor propagation events
- ✅ Labels, source tweets, and propagation trees

### Excluded (to optimize size):
- ❌ `.git/` directory
- ❌ `__pycache__/` and `*.pyc` files
- ❌ Previous `results/` and `checkpoints/`
- ❌ Virtual environments

---

## 🎯 Step-by-Step Deployment to Google Colab

### Step 1: Upload Files to Colab
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload **both** files to Colab:
   - Click the folder icon (📁) in the left sidebar
   - Click the upload button
   - Upload `colab_deployment.zip` (wait for upload to complete - ~12 MB)
   - The zip should appear in `/content/colab_deployment.zip`

### Step 2: Open the Notebook
1. In Colab, go to **File → Upload notebook**
2. Upload `Run_on_Colab.ipynb`
3. The notebook will open automatically

### Step 3: Configure Runtime (Important!)
1. Go to **Runtime → Change runtime type**
2. Select:
   - **Hardware accelerator:** GPU (T4 recommended)
   - **Runtime shape:** Standard
3. Click **Save**

### Step 4: Run the Pipeline
1. **Cell 1:** Mount Google Drive (authorize when prompted)
2. **Cell 2:** Extract deployment package
3. **Cell 3:** Install dependencies (~2-3 minutes)
4. **Cell 4:** Verify dataset integrity
5. **Cell 5:** Run TDA-Only experiment (~30-60 minutes)
6. **Cell 6:** Run Text-Only experiment (~20-40 minutes)
7. **Cell 7:** Run Hybrid experiment (~40-80 minutes)
8. **Cell 8:** Package results and save to Drive
9. **Cell 9:** Display final metrics summary

**Total Runtime:** ~2-3 hours (depending on GPU availability)

---

## 📊 Expected Outputs

### Results Directory Structure:
```
/content/project/results/
├── tda_only/
│   ├── test_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── classification_report.txt
├── text_only/
│   └── [same structure]
└── hybrid/
    └── [same structure]
```

### Checkpoints:
```
/content/project/checkpoints/
├── tda_only/
│   └── best_model.pt
├── text_only/
│   └── best_model.pt
└── hybrid/
    └── best_model.pt
```

### Google Drive Backup:
```
/content/drive/MyDrive/Project_Results/
└── rumor_detection_results_YYYYMMDD_HHMMSS.zip
```

---

## 🎓 Expected Performance Metrics

Based on previous experiments:

| Model       | Accuracy | F1 Score | Precision | Recall |
|-------------|----------|----------|-----------|--------|
| TDA-Only    | ~66%     | ~66%     | ~67%      | ~66%   |
| Text-Only   | ~90%     | ~90%     | ~90%      | ~90%   |
| **Hybrid**  | **~92%** | **~92%** | **~92%**  | **~92%**|

---

## 🔧 Troubleshooting

### Issue: "FileNotFoundError: colab_deployment.zip"
**Solution:** Ensure the zip file is uploaded to `/content/` (not `/content/sample_data/`)

### Issue: "CUDA out of memory"
**Solution:** 
- Reduce `batch_size` in config files (try 8 instead of 16)
- Use Runtime → Factory reset runtime
- Try again during off-peak hours

### Issue: "Dataset not found"
**Solution:** Run Cell 4 to verify. If failed, the zip extraction may be incomplete.

### Issue: Slow training
**Solution:** 
- Verify GPU is enabled (Runtime → Change runtime type)
- Check GPU usage: `!nvidia-smi`
- Reduce `max_events` in configs for faster testing

---

## 📝 Configuration Customization

### To modify experiment parameters:

Edit the config files before zipping, or modify in Colab after extraction:

```python
# In Colab, after Cell 2:
!nano /content/project/configs/colab_hybrid.yaml
```

**Key parameters:**
- `experiment.max_events`: Number of events to process (0 = all)
- `training.batch_size`: Batch size (reduce if OOM)
- `training.epochs`: Training epochs
- `training.learning_rate`: Learning rate
- `model.dropout`: Dropout rate for regularization

---

## 🎉 Success Indicators

You'll know everything worked when:
1. ✅ All 9 cells execute without errors
2. ✅ Cell 9 displays metrics for all 3 models
3. ✅ Google Drive contains the results zip file
4. ✅ Hybrid model F1 score > 90%

---

## 📧 Support

If you encounter issues:
1. Check the error message in the failed cell
2. Verify GPU is enabled
3. Ensure sufficient Drive storage (~500 MB)
4. Try reducing `batch_size` or `max_events`

---

## 🔄 Re-running Experiments

To run again with different parameters:
1. Modify config files in `/content/project/configs/`
2. Re-run cells 5-9
3. Previous results will be overwritten

---

**Created:** 2026-04-26  
**Package Size:** 12 MB (2,341 files)  
**Dataset:** ACL2017 (Twitter15 + Twitter16)  
**Models:** TDA-Only, Text-Only, Hybrid  
**Expected Runtime:** 2-3 hours on T4 GPU
