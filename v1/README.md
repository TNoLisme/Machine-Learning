# MALLORN TDE Classification - LightGBM Solution

Giải pháp phân loại Tidal Disruption Events (TDEs) từ simulated LSST lightcurves.

## Cấu trúc

```
v1/
├── 01_feature_engineering.ipynb   # Trích xuất features từ 20 splits
├── 02_model_training.ipynb        # Train LightGBM + hyperparameter tuning
├── 03_prediction_submission.ipynb # Tạo submission.csv (7135 rows)
├── requirements.txt
└── README.md
```

## Cách chạy

```bash
cd v1
pip install -r requirements.txt
```

**Chạy theo thứ tự:**

1. `01_feature_engineering.ipynb` - Xử lý TẤT CẢ 20 splits, tạo features
2. `02_model_training.ipynb` - Train model với hyperparameter search
3. `03_prediction_submission.ipynb` - Tạo file `submission.csv`

## Features (dựa trên paper insights)

### Per-band Features (6 bands × ~35 features)
- **Statistics**: mean, std, median, max, min, range, IQR, skew, kurtosis
- **Percentiles**: p10, p25, p75, p90
- **SNR**: mean, max, median, std
- **Temporal**: time_span, cadence, duration, peak_time, rise/decay rates
- **Variability**: coefficient of variation, trend slope, residuals

### Color Features (~20 features) - QUAN TRỌNG NHẤT
- **Colors**: u-g, u-r, u-i, g-r, g-i, r-i, i-z, z-y (mean + peak)
- **Blue indicators**: blue_fraction, u_fraction, g_fraction, blue_red_ratio
- **Peak band**: peak_band_is_u, peak_band_is_g, peak_band_is_blue

### Temporal Features (~10 features)
- total_time_span, detection_time_span
- peak_time_spread, peak_delay_u_r, peak_delay_g_r
- n_bands_detected

### Variability Features (~15 features)
- Per-band: flux_diff_std, flux_diff_mean, residual_std, cv
- Global: mean_variability, max_variability

### Metadata
- Redshift (Z), Extinction (EBV)

**Total: ~280 features**

## Model

### LightGBM Classifier
- Hyperparameter search với 54 combinations
- 5-fold Stratified Cross-Validation
- Class imbalance handling với `scale_pos_weight`
- Threshold optimization cho F1 score

### Key Parameters
```python
{
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [6, 8, 10],
    'num_leaves': [31, 63],
    'scale_pos_weight': ~15 (auto-calculated)
}
```

## Output Files

| File | Mô tả |
|------|-------|
| `train_features.csv` | Features cho training set |
| `test_features.csv` | Features cho test set |
| `lgb_model.joblib` | Trained LightGBM model |
| `scaler.joblib` | StandardScaler |
| `model_info.joblib` | Threshold, metrics, params |
| `feature_importance.csv` | Feature importances |
| `hyperparam_search_results.csv` | Kết quả hyperparameter search |
| **`submission.csv`** | File nộp Kaggle (7135 rows) |

## Key Insights từ Paper

1. **TDEs màu xanh**: Phát xạ mạnh ở u-band, color features quan trọng nhất
2. **Duration dài**: TDEs ~400+ days vs SNe ~100-150 days
3. **Smooth evolution**: TDEs ít variability hơn AGN
4. **Class imbalance**: ~6% TDEs, cần xử lý với scale_pos_weight
5. **Band importance**: u, g, r quan trọng nhất; z, y ít useful

## Metric

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

- Precision: Độ chính xác của predictions
- Recall: Khả năng tìm được tất cả TDEs

## Submission Format

```csv
object_id,prediction
Eluwaith_Mithrim_nothrim,0
Eru_heledir_archam,0
...
```

- `prediction`: 0 = Non-TDE, 1 = TDE
- **Phải có đúng 7135 rows!**

## Troubleshooting

1. **Thiếu rows trong submission**: Kiểm tra `01_feature_engineering.ipynb` đã chạy hết chưa
2. **Model không train được**: Cài đủ dependencies (`pip install -r requirements.txt`)
3. **Memory error**: Giảm số lượng hyperparameter combinations

## References

- Paper: Magill et al. 2025, "MALLORN: Many Artificial LSST Lightcurves based on Observations of Real Nuclear transients"
- Kaggle: https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge
