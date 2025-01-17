import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

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

    # 클래스별 데이터 분리
    class_groups = {
        30: df[df[angle_column] == 30],
        60: df[df[angle_column] == 60],
        90: df[df[angle_column] == 90],
        120: df[df[angle_column] == 120]
    }

    # Oversampling을 모든 클래스에 적용
    balanced_classes = {
        class_label: resample(class_data, replace=True, n_samples=1000, random_state=42)
        for class_label, class_data in class_groups.items()
    }

    # 최종 데이터셋 병합
    df_balanced = pd.concat(balanced_classes.values())

    # 균형 잡힌 데이터셋 저장
    balanced_csv_path = r"C:\Users\USER\balanced_training_data.csv"
    df_balanced.to_csv(balanced_csv_path, index=False)
    print(f"균형 잡힌 데이터셋이 저장되었습니다: {balanced_csv_path}")

    # 데이터 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(df_balanced[angle_column], bins=4, color='purple', alpha=0.7, edgecolor='black')
    plt.title("Balanced Steering Angle Distribution")
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
