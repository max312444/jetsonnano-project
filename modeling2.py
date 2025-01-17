import pandas as pd
import os

# 수정된 CSV 파일 경로
csv_path = r"C:\Users\USER\dataset_0117\lastdance_csv.csv"

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    print("CSV 파일이 정상적으로 로드되었습니다.")

    # 존재하지 않는 파일 확인
    missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

    # 결과 출력
    print(f"최종 누락된 파일 개수: {len(missing_files)}")
    if len(missing_files) > 0:
        print(f"누락된 파일 예시: {missing_files[:5]}")  # 누락된 파일 경로 일부 출력
    else:
        print("모든 파일이 정상적으로 존재합니다.")
except FileNotFoundError:
    print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
