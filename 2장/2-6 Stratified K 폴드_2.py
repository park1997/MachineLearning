from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import numpy as np


iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)
skfold = StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy=[]

# StratifiedKFold의 split() 호출시 반드시 레이블 데이터 셋도 추가 입력 필요
for train_index,test_index in skfold.split(features,label):
    X_train,X_test = features[train_index],features[test_index]
    y_train,y_test = label[train_index],label[test_index]

    # 학습 및 예측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)

    # 반복 시 마다 정확도 측정
    n_iter+=1
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("\n{} 교차 검증 정확도 : {}, 학습 데이터 크기  : {}, 검증 데이터 크기 : {} ".format(n_iter,accuracy,train_size,test_size))
    print("#{} 검증 세트 인덱스 : {}".format(n_iter,test_index))
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 계산
print("\n## 교차 검증별 정확도 : ",np.round(cv_accuracy,4))
print("## 평균 검증 정확도 : ",np.mean(cv_accuracy))

