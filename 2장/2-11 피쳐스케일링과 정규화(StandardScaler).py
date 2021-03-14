from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd


iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)

print("feature 들의 평균 값 ")
print(iris_df.mean())
print("\nfeature 들의 분산 값 ")
print(iris_data.var())

# StandardScaler 객체 생성
# StandardScaler : 평균이 0 이고, 분산이 1인 정규분포 형태로 변환
scaler = StandardScaler()
# StandardScaler로 데이터 셋 변환. fit()과 transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform() 시 scale 변환된 데이터 셋이 numpy ndarray로 반환되어 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
# 거의 0이라 0이라 봐도 무방함
print('feature 들의 평균 값')

print(iris_df_scaled.mean())
# 거의 1임. 1로 봐도 무방함.
print('\nfeature 들의 분산 값') 
print(iris_df_scaled.var())


