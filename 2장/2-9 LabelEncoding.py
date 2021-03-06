from sklearn.preprocessing import LabelEncoder

items=["TV","냉장고","전자렌지","컴퓨터","선풍기","선풍기","믹서","믹서"]

# LabelEncoder를 객체로 생성한 후 , fit() 과 transform() 으로 label 인코딩 수행.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print("인코딩 반환값 : ",labels)
print("인코딩 클래스 : ",encoder.classes_)
print("디코딩 원본 값 :",encoder.inverse_transform([0, 1, 4, 5, 3, 3, 2, 2]))
