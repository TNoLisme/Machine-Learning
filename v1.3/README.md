# MALLORN TDE Classification - v1.3

**Ensemble: LightGBM + XGBoost + CatBoost**

## Version History

| Version | CV F1 | Kaggle F1 | Gap | Notes |
|---------|-------|-----------|-----|-------|
| v1 | 0.68 | 0.5617 | 0.12 | Overfit |
| v1.1 | ? | 0.3989 | - | Underfit |
| v1.2 | 0.5972 | 0.5950 | 0.002 | Good generalization |
| v1.3 | ? | Target >0.62 | - | Ensemble boost |

## Strategy

v1.2 showed good generalization but low performance. v1.3 uses ensemble to boost:

1. **LightGBM** - Fast, good with tabular data
2. **XGBoost** - Strong regularization
3. **CatBoost** - Handles categorical well

## Ensemble Methods

1. **Simple Average**: (lgb + xgb + cat) / 3
2. **Weighted Average**: Based on individual CV F1 scores
3. **Stacking**: LogisticRegression meta-model
4. **Majority Voting**: 2/3 agreement

Best method is selected automatically based on CV performance.

## Usage

```bash
cd v1.3
pip install -r requirements.txt

jupyter notebook 01_feature_engineering.ipynb
jupyter notebook 02_model_training.ipynb
jupyter notebook 03_prediction_submission.ipynb
```

## Files

- `01_feature_engineering.ipynb` - Same as v1.2 (v1 + paper features)
- `02_model_training.ipynb` - Train LGB + XGB + CAT + ensemble
- `03_prediction_submission.ipynb` - Generate submission

## Output

- `lgb_model.joblib`, `xgb_model.joblib`, `cat_model.joblib`
- `stack_model.joblib` - Stacking meta-model
- `submission.csv` - Final predictions
- `full_predictions.csv` - With individual model probabilities
