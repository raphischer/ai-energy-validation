import cv2
import numpy as np
import sys
import os
import re

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == "__main__":
    images, labels = [], []
    for dirn in os.listdir('results'):
        if os.path.isdir(os.path.join('results', dirn)):
            for fname in tqdm(os.listdir(os.path.join('results', dirn))):
                matched = re.match(r'.*(\d)-(\d)(\d)(\d).jpg', fname)
                if matched:
                    img = cv2.imread(os.path.join('results', dirn, fname), cv2.IMREAD_GRAYSCALE)
                    for idx, (x0, x1) in enumerate( [[2, 27], [37, 62], [76, 101], [111, 136]] ):
                        images.append(img[10:50, x0:x1])
                        labels.append(matched.group(idx+1))
    
    images_np = np.array([img.flatten() for img in images])
    labels_np = np.array(labels)

    print('N DIGITS:', len(images))

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels_np)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in kf.split(images_np):
        X_train, X_test = images_np[train_idx], images_np[test_idx]
        y_train, y_test = labels_enc[train_idx], labels_enc[test_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Class-wise accuracy
        for class_idx in np.unique(labels_enc):
            mask = (y_test == class_idx)
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"Fold class {le.inverse_transform([class_idx])[0]} accuracy: {class_acc:.3f}")

    print(f"Mean 5-fold accuracy: {np.mean(accuracies):.3f}")

    final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    final_rf.fit(images_np, labels_enc)

    joblib.dump(final_rf, os.path.join('results', 'final_random_forest.pkl'))
