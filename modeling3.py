import pandas as pd
import os

# CSV 파일 경로
csv_path = r"C:\Users\USER\dataset_0117\lastdance_csv.csv"

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)
    print("CSV 파일이 정상적으로 로드되었습니다.")
    
    # 존재하지 않는 파일 확인
    missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]
    
    # 누락된 파일 개수 및 예시 출력
    print(f"누락된 파일 개수: {len(missing_files)}")
    if missing_files:
        print(f"누락된 파일 예시: {missing_files[:5]}")  # 최대 5개 예시 출력

    # 누락된 파일 제거
    df_cleaned = df[~df['frame_path'].isin(missing_files)]

    # 수정된 CSV 저장
    updated_csv_path = r"C:\Users\USER\dataset_0117\lastdance_csv.csv"
    df_cleaned.to_csv(updated_csv_path, index=False)
    print(f"수정된 CSV 파일이 저장되었습니다: {updated_csv_path}")

except FileNotFoundError:
    print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
