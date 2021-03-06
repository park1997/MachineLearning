from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import numpy as np
import pandas as pd

items=["TV","냉장고","전자렌지","컴퓨터","선풍기","선풍기","믹서","믹서"]

# 먼저 숫자값으로 변환을 위해 LabelEncoder로 변환한다.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# 2차원 데이터로 변환한다.
# 열은 무조건 하나, 행은 동적으로 변환
labels = labels.reshape(-1,1)

# One-Hot-Ecoding 진행한다.
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)

print("One-Hot-Encoding 데이터")
print(oh_labels.toarray())
print("One-Hot-Encoding 데이터 차원")
print(oh_labels.shape)

df = pd.DataFrame({"item":items})
print(df)
print(pd.get_dummies(df))
