import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import joblib

print("TUBERCULOSIS DETECTION SYSTEM")
print("=" * 60)

# =========================================================
# 1. LOAD DATASET (STEP-BY-STEP IMAGE COUNTING)
# =========================================================
print("\nSTEP 1: Loading Dataset (Sequential Mode)")

def load_images_stepwise(folder_path, label, target_size=(64, 64), log_every=100):
    images = []
    labels = []
    count = 0

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return images, labels

    file_list = os.listdir(folder_path)
    total_files = len(file_list)

    print(f"Loading from {folder_path}")
    print(f"Total images found: {total_files}")

    for idx, file in enumerate(file_list, start=1):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, target_size)
        img = img / 255.0
        img = img.flatten()

        images.append(img)
        labels.append(label)
        count += 1

        if idx % log_every == 0:
            print(f"Loaded {idx}/{total_files} images")

    print(f"Finished loading {count} images from {folder_path}")
    return images, labels


def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print("Dataset path not found")
        return None, None

    normal_path = os.path.join(dataset_path, "Normal")
    tb_path = os.path.join(dataset_path, "Tuberculosis")

    normal_images, normal_labels = load_images_stepwise(normal_path, 0)
    tb_images, tb_labels = load_images_stepwise(tb_path, 1)

    X = np.array(normal_images + tb_images)
    y = np.array(normal_labels + tb_labels)

    print("\nDataset Summary")
    print(f"Normal images: {len(normal_images)}")
    print(f"TB images: {len(tb_images)}")
    print(f"Total images: {len(X)}")

    return X, y


# Dataset path
dataset_path = r"C:\Users\AMAN12\Desktop\archive (4)\TB_Chest_Radiography_Database"
X, y = load_dataset(dataset_path)

if X is None or len(X) == 0:
    raise RuntimeError("Dataset loading failed")

# =========================================================
# 2. TRAIN-TEST SPLIT
# =========================================================
print("\nSTEP 2: Splitting Dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =========================================================
# 3. TRAIN MODEL
# =========================================================
print("\nSTEP 3: Training Random Forest Model")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training completed")

# =========================================================
# 4. EVALUATION
# =========================================================
print("\nSTEP 4: Evaluating Model")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * precision * sensitivity / (precision + sensitivity)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("\nPerformance Metrics")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Sensitivity  : {sensitivity:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC      : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ADD THESE TWO LINES
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)


# 5. VISUALIZATION
# =========================================================
# 5. VISUALIZATION (FIXED – SAME OUTPUT, NOW DISPLAYS)
# =========================================================
print("\nSTEP 5: Creating Medical Visualizations")

plt.figure(figsize=(15, 8))

# Confusion Matrix
plt.subplot(2, 3, 1)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted Normal', 'Predicted TB'],
    yticklabels=['Actual Normal', 'Actual TB']
)
plt.title("Confusion Matrix")

# ROC Curve
plt.subplot(2, 3, 2)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")

# Precision-Recall Curve
plt.subplot(2, 3, 3)
plt.plot(recall_curve, precision_curve, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")

# Metrics Bar Chart
plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
values = [accuracy, sensitivity, specificity, precision, f1]
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("Performance Metrics")

# Feature Importance
plt.subplot(2, 3, 5)
importances = model.feature_importances_
top_idx = np.argsort(importances)[-20:]
plt.barh(range(20), importances[top_idx])
plt.title("Top Feature Importances")

# Plot 6: Sample Image
plt.subplot(2, 3, 6)

sample_img = X[y == 0][0].reshape(64, 64)   # take first NORMAL image

plt.imshow(sample_img, cmap='gray')
plt.title("Sample Normal Chest X-ray")
plt.axis('off')


plt.tight_layout()
plt.savefig("medical_tb_analysis_comprehensive.png", dpi=130)

plt.show()


# =========================================================
# 6. SAVE MODEL
# =========================================================
print("\nSTEP 6: Saving Model")

joblib.dump(model, "tb_chest_xray_model.pkl")

print("Model saved as tb_chest_xray_model.pkl")

# =========================================================
# 7. USER IMAGE PREDICTION (ONE IMAGE AT A TIME)
# =========================================================
print("\nSTEP 7: Test Your Own X-ray Image")

def predict_single_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Unable to read image")
        return

    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten().reshape(1, -1)

    pred = model.predict(img)[0]
    prob = model.predict_proba(img)[0]

    label = "NORMAL" if pred == 0 else "TUBERCULOSIS"
    confidence = prob[pred]

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.3f}")

    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
    plt.title(label)
    plt.axis("off")
    plt.show()


while True:
    user_input = input("\nEnter image path (or 'quit'): ").strip()
    if user_input.lower() == "quit":
        break
    if not os.path.exists(user_input):
        print("File not found")
        continue
    predict_single_image(user_input)

print("\nProcess completed successfully")
