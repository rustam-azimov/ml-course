from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from adaboost import AdaBoost
from bagging import Bagging

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

bagging = Bagging(n_learners=10)
bagging.fit(X_train, y_train)

adaboost = AdaBoost(n_learners=10)
adaboost.fit(X_train, y_train)

tree_pred = tree.predict(X_test)
bagging_pred = bagging.predict(X_test)
adaboost_pred = adaboost.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_pred)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
adaboost_accuracy = accuracy_score(y_test, adaboost_pred)

print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print(f"Bagging Accuracy: {bagging_accuracy:.4f}")
print(f"AdaBoost Accuracy: {adaboost_accuracy:.4f}")
