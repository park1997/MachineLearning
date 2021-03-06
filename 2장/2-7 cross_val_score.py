from sklearn.datasets import load_iris
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)
data = iris_data.data
label = iris_data.target

# 성능 지표는 정확도(accuracy), 교차 검증 세트는 3개를 가짐
# DecisionTreeClassifier를 계속 학습하고 검증데이터로 평가하면서 정확도값을 내어주겠다!
scores = cross_val_score(dt_clf, data, label,scoring='accuracy',cv=3)
print("교차 검증별 정확도 : ",np.round(scores,4))
print("평균 검증 정확도 : ",np.round(np.mean(scores),4))


