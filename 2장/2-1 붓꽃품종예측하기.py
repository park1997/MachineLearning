import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 붓꽃 데이터 세트를 로딩합니다. 
iris = load_iris()

# iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다. 
iris_data = iris.data

# iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있습니다. 
iris_label = iris.target
# print('iris target값:', iris_label)
# print('iris target명:', iris.target_names)

# 붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다. 
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)
iris_df['label']=iris.target
# print(iris_df)
x_train , x_test, y_train, y_test = train_test_split(iris_data,iris_label,test_size =0.2, random_state=11)
#DecisonTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

# 학습수행
dt_clf.fit(x_train,y_train)
pred = dt_clf.predict(x_test)

#예측 값
print(pred)

print('예측 정확도 {:0.4}'.format(accuracy_score(y_test,pred)))
