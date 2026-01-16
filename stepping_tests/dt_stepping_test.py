import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('stepping_test_dataset_cleaned.csv')

df.drop(['Participant', 'TMD', 'MMD'], axis=1, inplace=True)

X = df.drop('Fall_Risk', axis=1)
y = df['Fall_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = DecisionTreeClassifier()
model = clf.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.3f}")

# Precision Score
precision = precision_score(y_test, y_pred, average='binary')
print(f"Precision: {precision:.3f}")

# F1-Score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.3f}")

y_score = model.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, y_score)
print(f"ROC-AUC: {roc_auc:.3f}")

confusion_matrix = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1])

cm_display.plot()
plt.show()

test_df = pd.DataFrame({'True': y_test, 'Decision Tree': y_pred})

plt.figure(figsize=(7, 5))

for model in ['Decision Tree']:
    fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()