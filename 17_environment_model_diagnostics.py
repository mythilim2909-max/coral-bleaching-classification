import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("./data/processed/coral_labels_with_environment.csv")
df = df.dropna(subset=['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk'])

feature_cols = ['max_dhw_trailing_12wk', 'mean_sst_trailing_12wk', 'mean_sst_anomaly_trailing_12wk']

# CHECK 1: How much real, distinct environmental information do we actually have?
unique_combos = df[feature_cols].drop_duplicates()
print(f"Total images: {len(df)}")
print(f"Unique environmental value combinations: {len(unique_combos)}")
print(f"(If this number is close to your site count ~112, the model is learning from far fewer truly distinct examples than the image count suggests)\n")

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']
X_train, X_test = train_df[feature_cols], test_df[feature_cols]
y_train = (train_df['label'] == 'bleached').astype(int)
y_test = (test_df['label'] == 'bleached').astype(int)

# CHECK 2: Is the EXISTING Random Forest underfitting or overfitting?
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest -- Train accuracy: {rf_train_acc*100:.2f}%  Test accuracy: {rf_test_acc*100:.2f}%")
if abs(rf_train_acc - rf_test_acc) < 0.05:
    print("  -> Train and test are close: this looks like UNDERFITTING or a genuine feature ceiling, not overfitting.\n")
else:
    print("  -> Train is notably higher than test: this looks like OVERFITTING.\n")

# CHECK 3: Try alternative models on the SAME features, SAME split -- fair comparison
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
print(f"Logistic Regression test accuracy: {accuracy_score(y_test, log_reg.predict(X_test))*100:.2f}%")

svm = SVC(kernel='rbf', class_weight='balanced')
svm.fit(X_train, y_train)
print(f"SVM (RBF kernel) test accuracy: {accuracy_score(y_test, svm.predict(X_test))*100:.2f}%")