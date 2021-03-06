# MinMaxScaler : 데이터값을 0과 1사이의 범위 값으로 변환합니다(음수 값이 있으면 -1 에서 1값으로 변환합니다.)

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)
scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print("feature들의 최소값")
print(iris_df_scaled.min())
print("feature들의 최대값")
print(iris_df_scaled.max())

# print(iris_df_scaled)