import pandas as pd
from sklearn.model_selection import train_test_split
import os

# CSV 파일 경로
csv_file_path = "C:/Users/USER/dataset_0117/data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_file_path)

# 이미지 파일명과 레이블을 분리 (예: 'image_name', 'angle', 'speed')
image_names = df['image_name'].values
angles = df['angle'].values
speeds = df['speed'].values

# 훈련 데이터와 테스트 데이터로 분리 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(image_names, 
                                                    pd.DataFrame({'angle': angles, 'speed': speeds}),
                                                    test_size=0.2, random_state=42)

# 분할된 데이터 확인
print("훈련 데이터 개수:", len(X_train))
print("테스트 데이터 개수:", len(X_test))

# 예시로 훈련 데이터와 테스트 데이터에서 첫 번째 항목 확인
print("훈련 데이터 예시 (이미지 파일명):", X_train[0])
print("훈련 데이터 예시 (각도, 속도):", y_train.iloc[0].to_dict())

print("테스트 데이터 예시 (이미지 파일명):", X_test[0])
print("테스트 데이터 예시 (각도, 속도):", y_test.iloc[0].to_dict())
