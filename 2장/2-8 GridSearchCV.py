from sklearn.datasets import load_iris
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# 데이터를 로딩하고 학습데이터와 데스트 데이터 분리
iris = load_iris()
iris_data = load_iris()
X_train, X_test, y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=121)

dtree = DecisionTreeClassifier()

# parameter 들을 dictionary형태로 저장
parameter = {"max_depth":[1,2,3],"min_samples_split":[2,3]}

# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정
# refit=True 가 default임. True 이면 가장 좋은 파라미터 설정으로 재 학습 시킴
