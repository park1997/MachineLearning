import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

iris_df = pd.DataFrame(data = iris.data, columns=iris.feature_names)
iris_df['label']=iris.target

kfold = KFold(n_splits=3)
# kfold.split(X)는 폴드 세트를 3번 반복할 때마다 달라지는 학습/테스트용 데이터 로우 인덱스 번호반환
n_iter=0
for train_index, test_index in kfold.split(iris_df):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print("## 교차 검증 : {}".format(n_iter))
    print("학습 레이블 데이터 분포 :\n",label_train.value_counts())
    print("검증 레이블 데이터 분포 :\n",label_test.value_counts())
    
print("-"*30)

# StratifiedKFold를 사용하면 레이블을 잘 균형해서 학습시킨다.
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

# StratifiedKFold는 split할때 레이블 인자값을 꼭 넣어줘야함
for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter+=1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(train_index,test_index)
    print("## 교차 검증 : {}".format(n_iter))
    print("학습 레이블 데이터 분포.\n",label_train.value_counts())
    print("검증 레이블 데이터 분포.\n",label_test.value_counts())




