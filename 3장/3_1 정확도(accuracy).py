import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits


class MyDummyClassifier(BaseEstimator):
    # fit() 메소드는 아무것도 학습하지 않음.
    def fit(self,X,y=None):
        pass

    #predict() 메소드는 단순히 sex feature가 1 이면 0 그렇지않으면 1로 예측함.
    def predict(self,X):
        pred = np.zeros((X.shape[0],1)) # 1은 2차원으로 명확하게 하겠다는건데 굳이 안써도 되긴함
        for i in range(X.shape[0]):
            if X["Sex"].iloc[i]==1:
                pred[i]=0
            else:
                pred[i]=1

        return pred
    
# null처리 함수
def fillna(df):
    df["Age"].fillna(df["Age"].mean(),inplace=True)
    df["Cabin"].fillna("N",inplace=True)
    df["Embarked"].fillna("N",inplace=True)
    df["Fare"].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행
def format_features(df):
    df["Cabin"] = df["Cabin"].str[:1]
    features = ["Cabin","Sex","Embarked"]
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 원본 데이터를 재로딩, 데이터 가공, 학습이터/테스트 데이터 분할
titanic_df = pd.read_csv("/Users/byeonghyeon/Documents/GitHub/MachineLearning/3장/train.csv")
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived",axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train,y_test = train_test_split(X_titanic_df,y_titanic_df, test_size=0.2,random_state = 11)

# 위에서 생성한 Dummy Classifier를 이용하여 학습/예측/평가 수행
myclf=MyDummyClassifier()
myclf.fit(X_train,y_train)

myprediction = myclf.predict(X_test)
print("Dummy Classifier의 정확도는 : {:.4f}".format(accuracy_score(y_test,myprediction)))



class MyFakeClassifier(BaseEstimator):
    def fit(self,X,y):
        pass

    # 입력값으로 들어오는 x 데이터 셋의 크기만큼 모두 0 값으로 만들어서 반환
    def predict(self,X):
        # 다 0으로 만듬
        return np.zeros((len(X),1),dtype=bool)

# 사이킷런의 내장 데이터 셋인 load_digits()를 이용하여 MIST 데이터 로딩
digits = load_digits()

print(digits.data)
print("### digits.data.shape",digits.data.shape)
print(digits.target)
print("### digits.target.shape",digits.target.shape)
print()

# digits번호가 7번이면 True이고 이를 astype(int)로 1로 변환, 7번이 아니면 False이고 0으로 변환
y=(digits.target==7).astype(int)
# print(y)
X_train,X_test,y_train,y_test = train_test_split(digits.data, y, random_state=11)

# 불균형한 레이블 데이터 분포도 확인
print("레이블 테스트 세트 크기 : ",y_test.shape)
print("테스트 세트 레이블 0 과 1 의 분포도")
print(pd.Series(y_test).value_counts())

# Dummy Classifier로 학습/예측/정확도 평가
fakeclf = MyFakeClassifier()
fakeclf.fit(X_train,y_train)
fakepred = fakeclf.predict(X_test)
print("모든 예측을 0으로 하여도 정확도는 {:.3f}".format(accuracy_score(y_test,fakepred)))


