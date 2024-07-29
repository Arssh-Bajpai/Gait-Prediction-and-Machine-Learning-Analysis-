from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

wine = datasets.load_wine()
X = wine['data']
y = wine['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

gbr = GradientBoostingClassifier()
gbr.fit(X_train, y_train)
print(cross_val_score(gbr, X_train, y_train, scoring='accuracy', cv=5))
param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 9]
}

from sklearn.model_selection import GridSearchCV
gbr2 = GridSearchCV(gbr, param_grid, cv = 3, n_jobs = 3)
gbr2.fit(X_train, y_train)
print(gbr2.best_params_)
print(gbr2.best_score_)







