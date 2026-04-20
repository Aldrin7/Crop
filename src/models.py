"""Classifier definitions — returns a dict of named estimators."""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from .config import RANDOM_STATE

def get_classifiers():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1),
        'SVM_RBF': SVC(
            kernel='rbf', C=10, gamma='scale',
            random_state=RANDOM_STATE, probability=True),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=15, min_samples_split=5, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(
            max_iter=5000, solver='lbfgs',
            C=1.0, random_state=RANDOM_STATE),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_STATE, batch_size=64),
        'GaussianNB': GaussianNB(),
    }

def get_optional_classifiers():
    """Returns XGBoost/LightGBM if installed, else empty dict."""
    extras = {}
    try:
        import xgboost as xgb
        extras['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss',
            use_label_encoder=False)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        extras['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    except ImportError:
        pass
    return extras

def all_classifiers():
    d = get_classifiers()
    d.update(get_optional_classifiers())
    return d
