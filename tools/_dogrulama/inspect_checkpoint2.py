import sys
sys.path.insert(0, r"C:\Users\Barış\Desktop\GBM-AID Prototip\gbm-aid mert")
import tools.train_cox_week3 as m
sys.modules['__main__'] = m  # pickle recorded classes under __main__

import pandas as pd
path = r"C:\Users\Barış\Desktop\GBM-AID Prototip\gbm-aid mert\artifacts\week3\cox_model\_checkpoints\arm_primary_wt93_icc60_th06.pkl"
obj = pd.read_pickle(path)
print("type:", type(obj))
print("name:", obj.name)
print("feature_columns (candidates):", len(obj.feature_columns))
fm = obj.final_model
print("final_features:", fm.final_features)
print("used_fallback:", fm.used_fallback)
print("best_penalizer:", fm.best_penalizer, "best_l1_ratio:", fm.best_l1_ratio)
cox = fm.fitted_model
print("fitted_model type:", type(cox))
print("cox params_ index:", list(cox.params_.index))
print("extra_columns:", getattr(obj, "extra_columns", "MISSING-ATTR"))
print("external_test:", obj.external_test)
