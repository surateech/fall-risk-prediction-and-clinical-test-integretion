import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

#===================================================================================================
# 1. Preprocessing Dataset
#===================================================================================================

df = pd.read_csv('stepping_test_dataset_cleaned.csv')

df.drop(['Participant', 'MMD', 'Total_Steps'], axis=1, inplace=True)

X = df.drop('Fall_Risk', axis=1)
y = df['Fall_Risk']

#===================================================================================================
# 2. Slicing Dataset (Training & Test Datasets)
#===================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#===================================================================================================
# 3. Training Model
#===================================================================================================

scaler = StandardScaler()
model = svm.SVC(kernel='linear', probability=True, C=1.0, gamma='scale', class_weight='balanced')

ovr_clf = OneVsRestClassifier(model)
ovo_clf = OneVsOneClassifier(model)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ovr_clf.fit(X_train_scaled, y_train)
ovo_clf.fit(X_train_scaled, y_train)

#===================================================================================================
# 4. Model Prediction
#===================================================================================================

y_pred_ovr = ovr_clf.predict(X_test_scaled)
y_score_ovr = ovr_clf.decision_function(X_test_scaled)

y_pred_ovo = ovo_clf.predict(X_test_scaled)
y_score_ovo = ovo_clf.decision_function(X_test_scaled)

#===================================================================================================
# 5. Model Evaluation
#===================================================================================================

# Accuracy Score
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)

print(f"OVR-Accuracy: {accuracy_ovr:.3f}")
print(f"OVO-Accuracy: {accuracy_ovo:.3f}")

# Recall Score
recall_ovr = recall_score(y_test, y_pred_ovr, labels=[0, 1, 2], average='micro')
recall_ovo = recall_score(y_test, y_pred_ovo, labels=[0, 1, 2], average='micro')

print(f"OVR-Recall: {recall_ovr:.3f}")
print(f"OVO-Recall: {recall_ovo:.3f}")

# Confusion Matrix
confusion_matrix_ovr = confusion_matrix(y_test, y_pred_ovr)

cm_display_ovr = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_ovr, display_labels = [0, 1, 2])

cm_display_ovr.plot()
plt.show()

confusion_matrix_ovo = confusion_matrix(y_test, y_pred_ovo)

cm_display_ovo = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_ovo, display_labels = [0, 1, 2])

cm_display_ovo.plot()
plt.show()

"""
#ROC-AUC Curve
test_df = pd.DataFrame({'True': y_test, 'Random Forest': y_pred})

plt.figure(figsize=(7, 5))

for model in ['Random Forest']:
    fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
"""