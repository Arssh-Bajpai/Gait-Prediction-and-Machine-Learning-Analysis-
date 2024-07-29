from sklearn import datasets

diabetes = datasets.load_diabetes()
print(type(diabetes), diabetes)

X = diabetes.data
y = diabetes.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("mean_absolute_error:", mean_absolute_error(y_test, y_pred))
print("mean_squared_error:", mean_squared_error(y_test, y_pred))
print("r2_score:", r2_score(y_test, y_pred))

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'criterion': ['friedman_mse']
}

from sklearn.model_selection import GridSearchCV

gbr_cv = GridSearchCV(gbr, param_grid, cv = 3, n_jobs = -1, scoring='neg_mean_squared_error')
gbr_cv.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
print("mean_absolute_error:", mean_absolute_error(y_test, y_pred))
print("mean_squared_error:", mean_squared_error(y_test, y_pred))
print("r2_score:", r2_score(y_test, y_pred))
print("best_params:", gbr_cv.best_params_)
print("best_score:", gbr_cv.best_score_)
print("best_estimator", gbr_cv.best_estimator_)

