import pandas as pd
import numpy as np

def load_har_data(train_path='data/train.csv', test_path='data/test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Séparer les colonnes
    feature_cols = [c for c in train.columns if c not in ('Activity', 'subject')]
    X_train = train[feature_cols].values.astype(np.float64)
    y_train = train['Activity'].values
    X_test = test[feature_cols].values.astype(np.float64)
    y_test = test['Activity'].values

    return X_train, y_train, X_test, y_test, feature_cols