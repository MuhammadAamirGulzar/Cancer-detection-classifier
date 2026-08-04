# Test Before Running Inference

## Run This First!

Before running the full inference, test your setup:

```bash
python test_inference_setup.py
```

## What It Tests

### ✅ Test 1: Base Directories
- Checks if TCGA models directory exists
- Checks if PAIP data directory exists  
- Checks if ground truth CSV exists

### ✅ Test 2: Ground Truth CSV
- Loads the CSV file
- Checks for required columns (WSI_Id, label/label_desc)
- Shows sample data

### ✅ Test 3: TCGA Model Files
- Checks all 12 model files (3 feature models × 4 classifiers)
- Shows which models are found/missing
- Displays file sizes

### ✅ Test 4: PAIP Feature Files
- Checks all 3 feature directories
- Counts .pt files in each
- Tests loading one file to verify format
- Shows feature dimensions

### ✅ Test 5: Model Loading
- Tests loading sklearn models (logistic, knn, protonet)
- Tests loading PyTorch ANN model
- Verifies models can be loaded without errors

### ✅ Test 6: Feature-Label Matching
- Compares WSI_Ids in features vs ground truth
- Shows how many samples will be processed
- Warns if no matches found

## Expected Output

If everything is correct, you'll see:

```
================================================================================
✅ ALL TESTS PASSED - Ready to run inference!
================================================================================

Next step: Run the main inference script
  python tcga_to_paip_inference.py
```

## If Tests Fail

The script will show exactly what's wrong:

- ❌ Missing directories
- ❌ Missing model files
- ❌ Missing feature files
- ❌ Mismatched WSI_Ids
- ❌ Wrong CSV structure

Fix the issues shown, then run the test again.

## Common Issues

### Issue: Model files not found
**Solution:** Check that models are in:
```
D:\...\TCGA_Result\5-Caption_based_aggregation\{feature_model}\MSIH\models\
```

### Issue: Feature files not found
**Solution:** Check that features are in:
```
D:\...\paip_data\slide_aggregation\Caption_Based_Clustering_FiveCrop\{feature_dir}\
```

### Issue: No matching samples
**Solution:** WSI_Ids in feature files must match WSI_Ids in ground truth CSV

### Issue: Wrong label column
**Solution:** CSV must have either 'label' or 'label_desc' column

## After Tests Pass

Once all tests pass, run the full inference:

```bash
python tcga_to_paip_inference.py
```

This will process all samples and save results to CSV.
