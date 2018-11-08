from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings

#to avoid FeatureWarning
warnings.filterwarnings("ignore")

def logistic_regression(x_train,y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    return model

def knn(x_train,y_train):
    model = KNeighborsClassifier()
    parameters = {"n_neighbors": np.arange(1, 30)}
    knn_cv = GridSearchCV(model, parameters, cv=5)
    knn_cv.fit(x_train, y_train)
    model = KNeighborsClassifier(n_neighbors=knn_cv.best_params_["n_neighbors"])
    model.fit(x_train, y_train)
    return model

def support_vector_machine(x_train,y_train):
    model = svm.SVC(kernel="linear")
    model.fit(x_train, y_train)
    return model

def random_forest(x_train,y_train):
    model = RandomForestClassifier(random_state=1, n_estimators=100)
    model.fit(x_train, y_train)
    return model
