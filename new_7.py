import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = r"C:\Users\USER\data.csv"

# CSV 파일 로드
try:
    df = pd.read_csv(csv_path)
    print("CSV 파일 열 이름:", df.columns.tolist())

    # 열 이름이 정확하지 않을 경우 대체 이름 확인
    if 'steering_angle' in df.columns:
        angle_column = 'steering_angle'
    elif 'angle' in df.columns:
        angle_column = 'angle'
    else:
        raise KeyError("CSV 파일에 'steering_angle' 또는 'angle' 열이 없습니다.")

    # 1. 특정 각도(-10, 10) 제거
    df_filtered = df[~df[angle_column].isin([-10, 10])]

    # 2. Oversampling: 가장 많은 데이터 수로 맞춤
    max_count = df_filtered[angle_column].value_counts().max()

    df_oversampled = df_filtered.groupby(angle_column, group_keys=False).apply(
        lambda x: x.sample(max_count, replace=True, random_state=42)
    )

    # 3. 결과 저장
    balanced_csv_path = r"C:\Users\USER\oversampled_training_data.csv"
    df_oversampled.to_csv(balanced_csv_path, index=False)
    print(f"Oversampled 데이터셋이 저장되었습니다: {balanced_csv_path}")

    # 4. 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(df_oversampled[angle_column], bins=len(df_oversampled[angle_column].unique()), 
             color='green', alpha=0.7, edgecolor='black')
    plt.title("Oversampled Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"CSV 파일이 존재하지 않습니다: {csv_path}")
except KeyError as e:
    print(f"열 이름 오류: {e}")
except Exception as e:
    print(f"예기치 않은 오류가 발생했습니다: {e}")
