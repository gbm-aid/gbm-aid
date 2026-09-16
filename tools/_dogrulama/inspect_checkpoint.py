import pandas as pd
import sys

path = r"C:\Users\Barış\Desktop\GBM-AID Prototip\gbm-aid mert\artifacts\week3\cox_model\_checkpoints\arm_primary_wt93_icc60_th06.pkl"
obj = pd.read_pickle(path)
print("type:", type(obj))
print("name:", obj.name)
print("feature_columns (candidates):", len(obj.feature_columns))
print("final_model type:", type(obj.final_model))
fm = obj.final_model
print("final_features:", fm.final_features)
print("used_fallback:", fm.used_fallback)
print("best_penalizer:", fm.best_penalizer, "best_l1_ratio:", fm.best_l1_ratio)
print("fitted_model type:", type(fm.fitted_model))
cox = fm.fitted_model
print("cox params_ index:", list(cox.params_.index))
print("cox summary columns present:", hasattr(cox, "summary"))
try:
    print(cox.print_summary())
except Exception as e:
    print("print_summary failed:", e)
print("extra_columns:", getattr(obj, "extra_columns", "MISSING-ATTR"))
print("external_test:", obj.external_test)
