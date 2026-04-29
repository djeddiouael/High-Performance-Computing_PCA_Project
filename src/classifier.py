from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import time

def train_and_evaluate(X_train, y_train, X_test, y_test, desc="SVM"):
    t0 = time.time()
    clf = LinearSVC(max_iter=2000, dual=False, random_state=42)
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{desc} – Temps entraînement: {train_time:.2f}s, Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return clf, acc