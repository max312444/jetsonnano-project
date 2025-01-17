import pandas as pd

# CSV 파일 경로 (데이터가 있는 CSV 파일 경로로 변경)
csv_file_path = "C:/Users/USER/dataset_0117/data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_file_path)

# 결측값 확인
print("결측값 확인:")
print(df.isnull().sum())  # 각 열의 결측값 수 확인

# 결측값 처리 (예: 평균값으로 채우기)
df_filled = df.fillna(df.mean())

# 결측값 처리 후 확인
print("\n결측값 처리 후:")
print(df_filled.isnull().sum())

# 처리된 데이터를 새 CSV 파일로 저장 (원본을 덮어쓰지 않도록)
df_filled.to_csv("C:/Users/USER/dataset_0117/data_filled.csv", index=False)

# 데이터의 첫 5개 행 출력 (처리된 데이터 확인)
print("\n처리된 데이터의 첫 5개 행:")
print(df_filled.head())
