from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier()

# 5개의 폴드세트로 분리하는 KFold객체와 폴드 세트별 정확도를 담을 리스트객체 생성.
kfold = KFold(n_splits=5, random_state=11)
cv_accuracy =[]

print("붓꽃 데이터 세트 크기 :",features.shape[0])

n_iter =0
# print(kfold.split(features))

# KFold 객체의 split() 호출하면 폴드 별 학습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features): # kfold.split(features)은 ndarray의 위치 인덱스를 반환해줌
    # kfold.split() 으로 반환된 인덱스를 이용하여 학습용, 검증용 데스트 데이터 추출
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    # print(train_index,test_index)
    #  학습 및 예측
    dt_clf.fit(x_train,y_train)
    pred = dt_clf.predict(x_test)
    n_iter +=1

    # 반복시 마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    print("\n {} 교차 검증 정확도 : {}, 학습 데이터 크기 : {}, 검증 데이터 크기 : {}".format(n_iter,accuracy,train_size,test_size))
    print(" {} 검증 세트 인덱스 : {}".format(n_iter,test_index))

    cv_accuracy.append(accuracy)

# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print("\n 평균 검증 정확도 : ", np.mean(cv_accuracy))



