# MALLORN TDE Classification Challenge

Giải pháp phân loại Tidal Disruption Events (TDEs) từ simulated LSST lightcurves.

## Bối cảnh khoa học

**Vera C. Rubin Observatory's LSST** sắp tạo ra một cuộc cách mạng trong thiên văn thời gian, dự kiến sẽ phát hiện số lượng transient nhiều hơn 100 lần so với hiện tại. Tuy nhiên, chúng ta không có đủ nguồn lực quang phổ để theo dõi tất cả các mục tiêu mà LSST sẽ phát hiện.

**Tidal Disruption Events (TDEs)** là một trong những transient hiếm và có giá trị khoa học cao nhất:
- Xảy ra khi ngôi sao đi quá gần hố đen siêu khối lượng và bị xé toạc
- Cung cấp phương pháp xác định mass hố đen ở dải thấp
- Đầu vào quan trọng cho vật lý bồi tụ và tiến hóa thiên hà
- Dự kiến LSST sẽ phát hiện vài nghìn TDE mỗi năm

**Thách thức chính:** Trong số hàng chục nghìn transient hàng năm, làm sao để xác định đâu là TDEs cần quan sát follow-up?

## Tổng quan

Dự án này thực hiện phân loại các sự kiện thiên văn để xác định đâu là Tidal Disruption Events (TDEs) và đâu là các sự kiện khác dựa trên dữ liệu lightcurve từ kính thiên văn LSST.

### Cấu trúc thư mục

````
ML/
├── v1/                    # LightGBM cơ bản
│   ├── 01_data_exploration.ipynb
│   ├── 01_feature_engineering.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_prediction_submission.ipynb
│   └── README.md
├── v1.2/                  # Cân bằng giữa comprehensive và regularization
│   ├── 01_feature_engineering.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_prediction_submission.ipynb
│   └── README.md
├── v1.3/                  # Ensemble models (LightGBM + XGBoost + CatBoost)
│   ├── 01_feature_engineering.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_prediction_submission.ipynb
│   └── README.md
└── v2/                    # Random Forest với hyperparameter tuning
    ├── 01_feature_engineering.ipynb
    ├── 02_model_training.ipynb
    ├── 03_prediction_submission.ipynb
    └── README.md
````

## Lịch sử phiên bản

| Version | Model | CV F1 | Kaggle F1 | Ghi chú |
|---------|-------|-------|-----------|---------|
| v1 | LightGBM | 0.5735 | 0.5617 | Overfitting (gap 0.12) |
| v1.2 | LightGBM | 0.5972 | 0.6027 | Good generalization (gap 0.005) |
| v1.3 | Ensemble | 0.6319 | 0.6385 | Majority voting (best performance) |
| v2 | Random Forest | ? | ? | OOB score, class balancing |

## Chiến lược chính

### 1. Feature Engineering

Dựa trên phân tích dữ liệu thực tế và đặc điểm vật lý của TDEs:

- **Color Features** (quan trọng nhất): TDEs có màu xanh (strong u-band)
  - Colors: u-g, u-r, u-i, g-r, g-i, r-i
  - Blue indicators: blue_fraction, u_fraction
  - Peak band features

- **Temporal Features**: TDEs kéo dài ~400+ ngày vs SNe ~100-150 ngày
  - time_span, detection_time_span, peak_time_spread
  - rise/decay rates, duration_class

- **Variability Features**: TDEs smooth hơn AGN
  - flux_diff_std, residual_std, autocorrelation
  - coefficient of variation

- **Per-band Statistics**: ~35 features/band
  - Statistics: mean, std, median, percentiles
  - SNR metrics, trend analysis
  - Detection kinetics: rise/decay rates, variability

- **Physics-guided Features** (v1.2): Based on TDE characteristics
  - Detection phase: pre/during/post peak counts
  - Redshift correction: color normalization
  - Duration & smoothness: autocorrelation, stochasticity

### 2. Modeling Approaches

#### v1 - v1.2: LightGBM
- Gradient boosting framework
- Class imbalance handling với `scale_pos_weight`
- Hyperparameter tuning
- Threshold optimization cho F1 score

#### v1.3: Ensemble Methods
- LightGBM + XGBoost + CatBoost
- Voting classifier và stacking
- Cross-validation cho từng model
- Ensemble optimization

#### v2: Random Forest
- Ensemble của decision trees
- Out-of-bag (OOB) score
- GridSearchCV cho hyperparameter tuning
- Feature importance analysis

### 3. Xử lý Class Imbalance

TDEs chỉ chiếm ~6% dataset:
- `scale_pos_weight` trong LightGBM
- `class_weight='balanced'` trong Random Forest
- Threshold optimization thay vì default 0.5
- F1 score làm metric chính

## Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd ML

# Cài đặt dependencies
pip install -r v1/requirements.txt
```

Dependencies chính:
- pandas, numpy
- scikit-learn
- lightgbm, xgboost, catboost
- matplotlib, seaborn
- scipy, tqdm
- joblib

## Cách sử dụng

### Chạy phiên bản v1.3 (recommended)

```bash
cd v1.3
jupyter notebook
```

Chạy theo thứ tự:
1. `01_feature_engineering.ipynb` - Trích xuất features từ 20 splits
2. `02_model_training.ipynb` - Train ensemble models
3. `03_prediction_submission.ipynb` - Tạo submission.csv

### Output files

- `train_features.csv` - Features cho training set
- `test_features.csv` - Features cho test set
- `submission.csv` - File nộp Kaggle (7135 rows)
- Models: `.joblib` files cho từng model
- Visualizations: feature importance, confusion matrix

## Kết quả

### Key Insights
1. **Color features quan trọng nhất**: u-r, u-g colors
2. **Duration matters**: TDEs kéo dài hơn SNe
3. **Blue dominance**: u-band là indicator mạnh nhất
4. **Regularization critical**: v1.2 achieved best generalization

### Performance Metrics
- **F1 Score**: Metric chính (balance precision & recall)
- **ROC-AUC**: ~0.94 cho hầu hết models
- **Precision**: ~0.6-0.7
- **Recall**: ~0.5-0.6

## Competition

- **Kaggle**: MALLORN Astronomical Classification Challenge
- **Dataset**: 20 splits của lightcurves
- **Target**: Binary classification (TDE vs Non-TDE)
- **Evaluation**: F1 Score

## Tài liệu tham khảo

1. TDE Characteristics in Optical Surveys
2. LSST Science Requirements  
3. Physics of Tidal Disruption Events

## Contributors

- Đào Ngọc Tân
- Nguyễn Văn Thịnh
- Bùi Hải Phương

---

**Lưu ý**: 
- Luôn chạy feature engineering trước khi training
- File submission phải có đúng 7135 rows
- Threshold optimization quan trọng cho F1 score
- Version v1.2 được recommend cho generalization tốt nhất
