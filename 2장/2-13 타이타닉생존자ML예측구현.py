# 데이터 전처리 => null처리, 불필요한 속성 제거, 인코딩 수행

# 모델 학습 및 검정/예측/평가 
# - 결정트리, 랜덤포레스트, 로지스틱 회귀 학습비교
# - KFold교차검증
# - cross_val_score(), GridSearchCV() 수행

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

titanic_df = pd.read_csv("/Users/byeonghyeon/Documents/GitHub/MachineLearning/2장/titanic/train.csv")
# print(titanic_df.head(3))

"""
Passengerid : 탑승자 데이터 일련번호
survived : 생존여부, 0 = 사망, 1 = 생존
Pclass : 티켓의 선실 등급, 1 = 일등성, 2= 이등석, 3 = 3등석
sex : 탑승자 성별
name : 탑승자 이름
Age : 탑승자 나이
sibap : 같이 탑승한 형제자매 또는 배우자 인원수
parch : 같이 탑승한 부모님 또는 어린이 인원수
ticket : 티켓 번호
fare : 요금
Cabin : 선실 번호
Embarked : 중간 정착 항구 C = Cherbourg, Q = Queenstown, S= Southampton
"""

print("#### train 데이터 정보 ####")
print(titanic_df.info())
print()

# NULL 컬럼들에대한 처리
titanic_df["Age"].fillna(titanic_df["Age"].mean(),inplace = True)
titanic_df["Cabin"].fillna("N",inplace = True)
titanic_df["Embarked"].fillna("N",inplace = True)

print("데이터 세트 NUll 값 갯수 ",titanic_df.isnull().sum().sum())
print()
print(" Sex 값 분포 :\n",titanic_df["Sex"].value_counts())
print()
print("\n Cabin 값 분포 :\n",titanic_df["Cabin"].value_counts())
print()
print("\n Embarked 값 분포 : \n",titanic_df["Embarked"].value_counts())
print()

titanic_df["Cabin"] = titanic_df["Cabin"].str[:1] # 셀안에 문자 빼올때는 str 함수 사용해야함!!
# print(titanic_df["Cabin"].head(3))

# 성별 별로 생존유무를 보여주기 
print(titanic_df.groupby(["Sex","Survived"])["Survived"].count())

sns.barplot(x="Sex",y="Survived",data=titanic_df)

sns.barplot(x="Pclass",y="Survived",hue="Sex",data=titanic_df)

# 입력 age에 따라 구분값을 반환하는 함수 설정. DataFrame의 apply lambda 식에 사용.
def get_category(age):
    cat=""
    if age<=-1:
        cat = "Unknown"
    elif age<=5:
        cat = "Baby"
    elif age<=12:
        cat = "Child"
    elif age<=18:
        cat = "Teenager"
    elif age<=25:
        cat = "Student"
    elif age<=35:
        cat = "Young Adult"
    elif age<=60:
        cat = "Adult"
    else:
        cat = "Elderyl"
    
    return cat

# 막대 그래프의 크기 figure을 더 크게 설정
plt.figure(figsize =(10,6))

# x축의 값을 순차적으로 표시하기 위한 설정
group_names = ["Unknown","Baby","Child","Teenager","Student","Young Adult","Adult","Elderly"]

# lambda 식에 위에서 생성한 get_category() 함수를 반환값으로 지정
# get_category(X)는 입력값으로 "Age" 컬럼값을 받아서 해당하는 cat 반환
titanic_df["Age_cat"] = titanic_df["Age"].apply(lambda x : get_category(x))
sns.barplot(x="Age_cat",y="Survived",hue="Sex",data=titanic_df,order=group_names)
print("\n",titanic_df["Age_cat"])
titanic_df.drop("Age_cat",axis=1,inplace=True) # axis=1 이라 열을 지움

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Lable 인코딩 진행
def encode_feature(dataDF):
    features = ["Cabin","Sex","Embarked"]
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    return dataDF

# null 처리 함수
def fillna(df):
    df["Age"].fillna(df["Age"].mean(),inplace=True)
    df["Cabin"].fillna("N",inplace = True)
    df["Embarked"].fillna("N",inplace = True)
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

# 앞에서 설정한 Data preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 원본 데이터를 재로딩 하고, feature데이터 셋과 Label 데이서 셋 추출
titanic_df = pd.read_csv("/Users/byeonghyeon/Documents/GitHub/MachineLearning/2장/titanic/train.csv")
y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived",axis=1)

X_titanic_df = transform_features(X_titanic_df)
# print(X_titanic_df)


X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size = 0.2,random_state= 11)

# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(max_iter=4000) # max_iter가 작으면 converge하지 못함 4000으로 늘려주자 !

# DecisonTreeClassifier 학습/예측/평가
dt_clf.fit(X_train,y_train)
dt_pred = dt_clf.predict(X_test)
print("\nDecisionTreeClassifier 정확도 : {0:.4f}".format(accuracy_score(y_test,dt_pred)))

# RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train,y_train)
rf_pred = rf_clf.predict(X_test)
print("RandomForestClassifier 정확도 : {0:.4f}".format(accuracy_score(y_test,rf_pred)))

# LogisticRegression 학습/예측/평가
lr_clf.fit(X_train,y_train)
lr_pred = lr_clf.predict(X_test)
print("LogisticRegression 정확도 : {0:.4f}".format(accuracy_score(y_test,lr_pred)))


def exec_kfold(clf,folds=5):
    # 폴드세트를 5개인 KFold객체를 생성, 폴드 수만큼 예측결과 저장을 위한 리스트 객체 생성

