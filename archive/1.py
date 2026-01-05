import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = r"C:\Users\USER\jetson_csv\steering_data.csv"

# 1. CSV 파일 로드
df = pd.read_csv(csv_path)

# 2. 'angle' 열을 사용하여 필터링
df_filtered = df[(df['angle'] != -10) & (df['angle'] != 10)]
print(f"필터링 후 데이터프레임 크기: {df_filtered.shape}")

# 3. Oversampling
# 각 각도의 데이터 개수를 확인
angle_counts = df_filtered['angle'].value_counts()
print("각도별 데이터 분포:\n", angle_counts)

# Oversampling 기준: 가장 많은 데이터(최대 빈도)의 개수를 기준으로 맞춤
max_count = angle_counts.max()

# 각 각도에 대해 Oversampling 수행
df_oversampled = df_filtered.groupby('angle', group_keys=False).apply(
    lambda x: x.sample(max_count, replace=True, random_state=42)
)
print(f"Oversampling 후 데이터프레임 크기: {df_oversampled.shape}")

# 4. **direction 열 삭제 (필요한 경우)**
if 'direction' in df_oversampled.columns:
    df_oversampled = df_oversampled.drop(columns=['direction'])
    print("'direction' 열이 삭제되었습니다.")

# 5. 결과 저장
balanced_csv_path = r"C:\Users\USER\jetson_csv\steering_data_balanced.csv"
df_oversampled.to_csv(balanced_csv_path, index=False)
print(f"Oversampled 데이터셋이 저장되었습니다: {balanced_csv_path}")

# 6. 새로운 분포 시각화
plt.figure(figsize=(10, 6))
plt.hist(df_oversampled['angle'], bins=len(df_oversampled['angle'].unique()), color='green', alpha=0.7, edgecolor='black')
plt.title("Oversampled Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
