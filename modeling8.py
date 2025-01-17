import pandas as pd
import os

# 기존 CSV 파일 경로
existing_csv_path = r"C:\Users\USER\dataset_0117\lastdance_csv.csv"  # 기존 파일 경로

# 새로운 CSV 파일 경로
new_csv_path = r"C:\Users\USER\new_csv"  # 새로 저장할 파일 경로

# 기존 파일에서 새 파일로 저장
try:
    # 기존 CSV 파일 로드
    df = pd.read_csv(existing_csv_path)

    # 새로운 CSV 파일로 저장
    os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)  # 새로운 파일 경로의 폴더가 없으면 생성
    df.to_csv(new_csv_path, index=False)
    print(f"새로운 CSV 파일이 생성되었습니다: {new_csv_path}")
except FileNotFoundError:
    print(f"기존 CSV 파일을 찾을 수 없습니다: {existing_csv_path}")
except Exception as e:
    print(f"오류 발생: {e}")
