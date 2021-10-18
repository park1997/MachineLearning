# GridSearchCV -> 교차 검증과 최적 하이퍼파라미터 튜닝을 한번에!!
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
# for문을 쓰면서 하이어파라미터를 바꿔주는경우 코드의 가시성이나 시간이 오래걸리게됨
# 따라서 이렇게 파라미터를 딕셔너리의 형태로 하이퍼파라미터 name지정후
# 이에따라 순차적으로 수행해야될 하이퍼파라미터 값을 리스트 형태로 넣어줌 
parameter = {"max_depth":[1,2,3],"min_samples_split":[2,3]}

# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정
# param_grid 는 딕셔너리 형태의 파라미터들을 받음
# (key값 = 하이퍼파라미터 이름), (value값 : "리스트"형태의 값)
# cv = 3 dms 3세트로 나누겠다는 뜻
# refit=True 가 default임. True 이면 가장 좋은 파라미터 설정으로 재 학습 시킴(가장 좋은 결과를 도출할 수 있도록 함)
grid_dtree = GridSearchCV(dtree,param_grid = parameter,cv=3,refit=True,return_train_score=True)

# 붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가.
grid_dtree.fit(X_train,y_train)

# GridSearchCV 결과는 cv_results_ 라는 딕셔너리로 저장됨. 이를 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[["params","mean_test_score","rank_test_score","split0_test_score","split1_test_score","split2_test_score"]]

print(scores_df[["mean_test_score","params"]])

print("GridSearchCV 최적 파라미터 : ",grid_dtree.best_params_) # 최적의 파라미터를 알려줌
print("GridSearchCV 최고 정확도 : {0:.4f}".format(grid_dtree.best_score_))

# refit = True로 설정된 GridSearchCV 객체가 fit()을 수행시 학습이 완료된 Estimator를 내포하고 있으므로 predict()를 통해 예측도 가능
pred = grid_dtree.predict(X_test)

print("테스트 데이터 세트 정확도 : {0:.4f}".format(accuracy_score(y_test,pred)))

# GridSearchCV의 refit으로 이미 학습이 된 estimator 반환
estimator = grid_dtree.best_estimator_

# GridSearchCV의 best_estimator_ 는 이미 최적 하이퍼 파라미터로 학습이 됨.
pred = estimator.predict(X_test)
print("테스트 데이터 세트 정확도 : {0:.4f}".format(accuracy_score(y_test,pred)))


