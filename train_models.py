import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                              matthews_corrcoef, roc_auc_score, confusion_matrix)

os.makedirs("results/models", exist_ok=True)

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
X_train, y_train = train[["PC1","PC2"]].values, train["Label"].values
X_test,  y_test  = test[["PC1","PC2"]].values,  test["Label"].values

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

MODELS = {
    "kNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=1))]),
    "RF" : RandomForestClassifier(class_weight="balanced", random_state=1),
    "NB" : GaussianNB(),
    "GB" : GradientBoostingClassifier(random_state=1),
}

PARAM_GRIDS = {
    "kNN": {"clf__n_neighbors": [5,10,15], "clf__weights": ["uniform","distance"]},
    "SVM": {"clf__C": [0.1,1,10], "clf__gamma": ["scale","auto"]},
    "RF" : {"n_estimators":[100,200], "max_depth":[None,5,10],
            "min_samples_split":[2,5], "min_samples_leaf":[1,2], "max_features":["sqrt","log2"]},
    "NB" : {},
    "GB" : {"n_estimators":[100,200], "learning_rate":[0.05,0.1], "max_depth":[3,5]},
}

def metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    return {
        "Accuracy"   : round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Specificity": round(tn/(tn+fp) if (tn+fp)>0 else 0, 4),
        "F1"         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC"        : round(matthews_corrcoef(y_true, y_pred), 4),
        "AUC"        : round(roc_auc_score(y_true, y_prob), 4),
    }

train_rows, test_rows = [], []

for name, model in MODELS.items():
    grid = PARAM_GRIDS[name]
    if grid:
        gs   = GridSearchCV(model, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        print(f"{name} best params: {gs.best_params_}")
    else:
        best = model
        best.fit(X_train, y_train)

    y_cv_pred = cross_val_predict(best, X_train, y_train, cv=cv)
    y_cv_prob = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:,1]
    tr = metrics(y_train, y_cv_pred, y_cv_prob); tr["Model"] = name
    train_rows.append(tr)

    y_te_pred = best.predict(X_test)
    y_te_prob = best.predict_proba(X_test)[:,1]
    te = metrics(y_test, y_te_pred, y_te_prob); te["Model"] = name
    test_rows.append(te)

    joblib.dump(best, f"{name}_model.pkl")
    print(f"{name} -> Test AUC={te['AUC']}  MCC={te['MCC']}  Acc={te['Accuracy']}")

cols = ["Model","Accuracy","Sensitivity","Specificity","F1","MCC","AUC"]
pd.DataFrame(train_rows)[cols].to_csv("train.csv", index=False)
pd.DataFrame(test_rows)[cols].to_csv("test.csv",  index=False)
print("\nAll models saved.")
