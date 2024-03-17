import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from bagging import Bagging
from adaboost import AdaBoost
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

df = pd.read_csv('CarsData.csv')
target = 'Manufacturer'
features = df.drop(columns=[target])

categorical_features = features.select_dtypes(include=['object', 'category']).columns
features = pd.get_dummies(features, columns=categorical_features)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target])

scaler = StandardScaler()
X = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


def optimize(trial):
    classifier = trial.suggest_categorical('classifier',
                                           ['RandomForest', 'AdaBoost', 'GradientBoosting', 'HistGradientBoosting',
                                            'XGBoost', 'LightGBM', 'CatBoost', 'Bagging'])

    if classifier == 'RandomForest':
        n_estimators = trial.suggest_int('rf_n_estimators', 10, 50)
        max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    elif classifier == 'Bagging':
        n_learners = trial.suggest_int('bagging_n_learners', 5, 20)
        clf = Bagging(n_learners=n_learners)

    elif classifier == 'AdaBoost':
        n_learners = trial.suggest_int('adaboost_n_learners', 5, 20)
        clf = AdaBoost(n_learners=n_learners)

    elif classifier == 'GradientBoosting':
        n_estimators = trial.suggest_int('gb_n_estimators', 10, 50)
        learning_rate = trial.suggest_loguniform('gb_learning_rate', 1e-3, 1e0)
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    elif classifier == 'HistGradientBoosting':
        max_iter = trial.suggest_int('hgb_max_iter', 10, 100)
        learning_rate = trial.suggest_loguniform('hgb_learning_rate', 1e-3, 1e0)
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate, random_state=42)

    elif classifier == 'XGBoost':
        n_estimators = trial.suggest_int('xgb_n_estimators', 10, 50)
        max_depth = trial.suggest_int('xgb_max_depth', 2, 10)
        learning_rate = trial.suggest_loguniform('xgb_learning_rate', 1e-3, 1e0)
        clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                use_label_encoder=False, eval_metric='logloss', random_state=42)

    elif classifier == 'LightGBM':
        n_estimators = trial.suggest_int('lgb_n_estimators', 10, 50)
        num_leaves = trial.suggest_int('lgb_num_leaves', 2, 256)
        learning_rate = trial.suggest_loguniform('lgb_learning_rate', 1e-3, 1e0)
        clf = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
                                 random_state=42)

    else:  # CatBoost
        iterations = trial.suggest_int('cb_iterations', 10, 50)
        depth = trial.suggest_int('cb_depth', 2, 10)
        learning_rate = trial.suggest_loguniform('cb_learning_rate', 1e-3, 1e0)
        clf = cb.CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, verbose=0,
                                    random_state=42)

    return cross_val_score(clf, X_train, y_train, n_jobs=-1, cv=3).mean()


study = optuna.create_study(direction='maximize')
study.optimize(optimize, n_trials=10)

best_params = study.best_trial.params
best_classifier = best_params['classifier']

model_map = {
    'RandomForest': lambda params: RandomForestClassifier(
        **{k.replace("rf_", ""): v for k, v in params.items() if k.startswith("rf_")}),
    'GradientBoosting': lambda params: GradientBoostingClassifier(
        **{k.replace("gb_", ""): v for k, v in params.items() if k.startswith("gb_")}),
    'HistGradientBoosting': lambda params: HistGradientBoostingClassifier(
        **{k.replace("hgb_", ""): v for k, v in params.items() if k.startswith("hgb_")}),
    'XGBoost': lambda params: xgb.XGBClassifier(
        **{k.replace("xgb_", ""): v for k, v in params.items() if k.startswith("xgb_")}, use_label_encoder=False,
        eval_metric='logloss'),
    'LightGBM': lambda params: lgb.LGBMClassifier(
        **{k.replace("lgb_", ""): v for k, v in params.items() if k.startswith("lgb_")}),
    'CatBoost': lambda params: cb.CatBoostClassifier(
        **{k.replace("cb_", ""): v for k, v in params.items() if k.startswith("cb_")}, verbose=0),
    'Bagging': lambda params: Bagging(n_learners=params.get('bagging_n_learners', 10)),
    'AdaBoost': lambda params: AdaBoost(n_learners=params.get('adaboost_n_learners', 10))
}

if best_classifier in model_map:
    model = model_map[best_classifier](best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f'Best classifier: {best_classifier}')
    print(f'Accuracy of the best classifier after hyperparameter optimization: {accuracy:.4f}')

    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC Score: {roc_auc:.4f}')

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

else:
    print(f'Classifier {best_classifier} is not recognized.')
