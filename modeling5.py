import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV 파일 경로
csv_path = r"C:\Users\USER\jetson_csv\steering_data.csv"

try:
    # 파일 경로 확인
    if not os.path.exists(csv_path):
        print(f"파일을 찾을 수 없습니다: {csv_path}")
    else:
        # CSV 파일 로드
        df = pd.read_csv(csv_path)

        # CSV 파일 내 열 확인
        print("CSV 열 목록:", df.columns)

        # `steering_angle` 데이터 확인 및 처리
        if 'steering_angle' in df.columns:
            df['steering_angle'] = pd.to_numeric(df['steering_angle'], errors='coerce')  # 숫자로 변환
            df = df.dropna(subset=['steering_angle'])  # NaN 값 제거

            if df.empty:
                print("데이터프레임이 비어 있습니다. CSV 파일을 확인하세요.")
            else:
                # 데이터 시각화를 위해 조향각 분포 확인
                plt.figure(figsize=(10, 6))
                plt.hist(df['steering_angle'], bins=30, color='blue', alpha=0.7, edgecolor='black')
                plt.title("Steering Angle Distribution")
                plt.xlabel("Steering Angle")
                plt.ylabel("Frequency")
                plt.grid(True)

                # 그래프 보여주기
                plt.show()
        else:
            print("CSV 파일에 'steering_angle' 열이 없습니다.")

except Exception as e:
    print(f"오류 발생: {e}")
