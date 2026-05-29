import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.model.model_preparation import (
    train_features, train_target,
    val_features,   val_target,
    test_features,  test_target,
)

# ── Scale features (fit only on train to avoid leakage) ──────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_val   = scaler.transform(val_features)
X_test  = scaler.transform(test_features)

y_train = train_target.values
y_val   = val_target.values
y_test  = test_target.values


# ── Tune alpha on validation set ──────────────────────────────────────
alphas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
best_alpha, best_val_mae = None, float("inf")

print("Alpha tuning on validation set:")
print(f"{'alpha':>10}  {'val MAE':>10}  {'val RMSE':>10}")
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mae  = mean_absolute_error(y_val, preds)
    rmse = root_mean_squared_error(y_val, preds)
    print(f"{alpha:>10.2f}  {mae:>10.3f}  {rmse:>10.3f}")
    if mae < best_val_mae:
        best_val_mae = mae
        best_alpha   = alpha

print(f"\nBest alpha: {best_alpha}  (val MAE={best_val_mae:.3f})")


# ── Retrain on train+val with best alpha ──────────────────────────────
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

final_model = Ridge(alpha=best_alpha)
final_model.fit(X_trainval, y_trainval)


# ── Evaluate on held-out test set ─────────────────────────────────────
y_pred = final_model.predict(X_test)
test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)
baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_trainval.mean()))

print(f"\nTest set results (alpha={best_alpha}):")
print(f"  MAE        : {test_mae:.3f}")
print(f"  RMSE       : {test_rmse:.3f}")
print(f"  Baseline MAE (predict mean): {baseline_mae:.3f}")


# ── Coefficients ranked by importance ────────────────────────────────
coef_df = pd.DataFrame({
    "feature": train_features.columns,
    "coefficient": final_model.coef_,
}).assign(abs_coef=lambda d: d["coefficient"].abs()) \
  .sort_values("abs_coef", ascending=False) \
  .drop(columns="abs_coef")

print("\nFeature coefficients (sorted by magnitude):")
print(coef_df.to_string(index=False))
