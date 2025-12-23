# MALLORN TDE Classification - Random Forest Solution (v2)

Giải pháp Random Forest cho phân loại Tidal Disruption Events (TDEs).

## Cấu trúc

```
v2/
├── 01_feature_engineering.ipynb   # Trích xuất features từ 20 splits
├── 02_model_training.ipynb        # Train Random Forest + GridSearchCV
├── 03_prediction_submission.ipynb # Tạo submission.csv (7135 rows)
├── requirements.txt
└── README.md
```

## Cách chạy

```bash
cd v2
pip install -r requirements.txt
```

**Chạy theo thứ tự:**

1. `01_feature_engineering.ipynb` - Xử lý TẤT CẢ 20 splits
2. `02_model_training.ipynb` - Train Random Forest với GridSearchCV
3. `03_prediction_submission.ipynb` - Tạo file `submission.csv`

## Model: Random Forest

### Đặc điểm
- **Ensemble method**: Kết hợp nhiều decision trees
- **Class balancing**: `class_weight='balanced'`
- **OOB Score**: Out-of-bag score để đánh giá không cần validation set
- **Feature importance**: Dựa trên Gini impurity

### Hyperparameter Search (GridSearchCV)
```python
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

**Total: 216 combinations** với 5-fold CV

### Ưu điểm của Random Forest
1. **Robust**: Ít bị overfitting hơn single decision tree
2. **No scaling needed**: Nhưng vẫn scale để consistency
3. **Feature importance**: Built-in feature importance
4. **OOB Score**: Estimate generalization error
5. **Handles imbalance**: `class_weight='balanced'`

## Features (~280 features)

### Per-band Features (6 bands × ~35 features)
- Statistics: mean, std, median, max, min, range, IQR, skew, kurtosis
- SNR: mean, max, median, std
- Temporal: duration, cadence, rise/decay rates
- Variability: cv, trend slope, residuals

### Color Features (~20 features) - QUAN TRỌNG
- Colors: u-g, u-r, u-i, g-r (mean + peak)
- Blue indicators: blue_fraction, u_fraction, blue_red_ratio
- Peak band indicators

### Temporal & Variability Features (~25 features)
- Cross-band temporal correlations
- Variability metrics

## Output Files

| File | Mô tả |
|------|-------|
| `train_features.csv` | Features cho training set |
| `test_features.csv` | Features cho test set |
| `rf_model.joblib` | Trained Random Forest model |
| `scaler.joblib` | StandardScaler |
| `model_info.joblib` | Threshold, metrics, params |
| `feature_importance.csv` | Feature importances |
| `hyperparam_search_results.csv` | GridSearchCV results |
| **`submission.csv`** | File nộp Kaggle (7135 rows) |

## So sánh với v1 (LightGBM)

| Aspect | v1 (LightGBM) | v2 (Random Forest) |
|--------|---------------|-------------------|
| Model type | Gradient Boosting | Bagging |
| Training speed | Nhanh hơn | Chậm hơn |
| Hyperparameter | 54 combos | 216 combos |
| Class balance | scale_pos_weight | class_weight='balanced' |
| Extra metric | - | OOB Score |

## Submission Format

```csv
object_id,prediction
Eluwaith_Mithrim_nothrim,0
Eru_heledir_archam,0
...
```

**Phải có đúng 7135 rows!**

## References

- Paper: Magill et al. 2025, "MALLORN"
- Kaggle: https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge
