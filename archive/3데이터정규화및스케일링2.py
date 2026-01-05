import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# CSV 파일 경로
csv_file_path = "C:/Users/USER/dataset_0117/data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_file_path)

# 센서 데이터 예시 (각도, 속도 컬럼이 있다고 가정)
# 각도와 속도 컬럼 정규화
scaler = MinMaxScaler()

# 각도와 속도 컬럼만 선택 (예시)
sensor_data = df[['angle', 'speed']]  # 'angle', 'speed' 컬럼이 있다고 가정

# 센서 데이터 정규화 (0~1 범위로 변환)
sensor_data_normalized = scaler.fit_transform(sensor_data)

# 정규화된 센서 데이터 DataFrame으로 변환
df_normalized = pd.DataFrame(sensor_data_normalized, columns=['angle', 'speed'])

# 원본 DataFrame에 정규화된 데이터를 합침
df[['angle', 'speed']] = df_normalized

# 정규화된 데이터 확인
print(df.head())

# 정규화된 데이터를 새로운 CSV 파일로 저장
df.to_csv("C:/Users/USER/dataset_0117/data_normalized.csv", index=False)
