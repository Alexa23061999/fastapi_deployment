import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
x = iris.data
y = iris.target

rf = RandomForestClassifier().fit(x, y)

with open('./api/mlmodel.pkl', 'wb') as f:
    pickle.dump(rf, f)














# import joblib
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier

# iris = load_iris()

# x = iris.data
# y = iris.target


# rf = RandomForestClassifier().fit(x, y)

# joblib.dump(rf, './api/mlmodel.joblib')
