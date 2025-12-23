# MALLORN TDE Classification - v1.2

**Balanced approach: v1 features + paper insights + moderate regularization**

## Version History

| Version | CV F1 | Kaggle F1 | Issue |
|---------|-------|-----------|-------|
| v1 | 0.68 | 0.5617 | Overfitting (gap 0.12) |
| v1.1 | ? | 0.3989 | Underfitting (too conservative) |
| v1.2 | ? | Target >0.58 | Balanced approach |

## v1.2 Strategy

1. **Keep ALL v1 features** (~280) - don't reduce
2. **Add paper-guided features** (~15 new) - detection phases, z-correction, smoothness
3. **Moderate regularization** - between v1 and v1.1

## Hyperparameter Comparison

| Param | v1 | v1.1 | v1.2 |
|-------|-----|------|------|
| n_estimators | 300-800 | 100-300 | **200-500** |
| max_depth | 6-10 | 3-5 | **5-7** |
| num_leaves | 31-63 | 7-31 | **15-31** |
| min_child_samples | 20 | 50 | **30** |
| reg_alpha | 0.1 | 0.5 | **0.2** |
| reg_lambda | 0.1 | 1.0 | **0.3** |
| subsample | 0.8 | 0.7 | **0.8** |
| colsample_bytree | 0.8 | 0.6 | **0.8** |

## New Paper-Guided Features (Added)

### Detection Phase Features
- `n_det_pre_peak` - Detections before peak
- `n_det_near_peak` - Detections within ±10 days of peak
- `n_det_post_peak` - Detections 10-30 days after peak
- `n_bands_near_peak` - Multi-band coverage near peak
- `has_u_near_peak` - Critical u-band near peak
- `some_color_score` - Paper's detectability metric

### Redshift-Corrected Features
- `color_u_r_z_norm` - u-r color normalized by (1+z)

### Duration & Smoothness
- `duration_class` - 0=short (SN), 1=medium, 2=long (TDE)
- `g_autocorr`, `r_autocorr` - Flux autocorrelation
- `g_sign_change_rate`, `r_sign_change_rate` - Stochasticity

## Usage

```bash
cd v1.2
pip install -r requirements.txt

# Run in order:
jupyter notebook 01_feature_engineering.ipynb
jupyter notebook 02_model_training.ipynb
jupyter notebook 03_prediction_submission.ipynb
```

## Files

- `01_feature_engineering.ipynb` - v1 features + paper additions
- `02_model_training.ipynb` - Balanced LightGBM training
- `03_prediction_submission.ipynb` - Generate submission.csv
- `requirements.txt` - Dependencies
- `README.md` - This file
