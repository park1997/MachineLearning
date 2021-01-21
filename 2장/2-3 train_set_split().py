import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris_data = load_iris()
dt_clf = DecisionTreeClassifier()

x_train,x_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.3,random_state=121)

dt_clf.fit(x_train,y_train)
pred= dt_clf.predict(x_test)

print("예측 정확도 : {:.4}".format(accuracy_score(y_test,pred)))


iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df["target"]=iris_data.target
#print(iris_df.head(10))

# 라벨을 제외한 피쳐데이터들
ftr_df = iris_df.iloc[:,:-1]
# 라벨 데이터들
tgt_df = iris_df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(ftr_df,tgt_df, test_size = 0.3, random_state= 121)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)

pred = dt_clf.predict(x_test)

print("예측 정확도 : {:.4}".format(accuracy_score(y_test,pred)))



