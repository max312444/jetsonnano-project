import pandas as pd
import os

# CSV 파일 경로 및 기본 경로
base_path = r"C:\Users\USER\dataset_0117"
csv_file_path = r"C:\Users\USER\data.csv"

# CSV 파일 로드
try:
    df = pd.read_csv(csv_file_path)
    print("CSV 파일 로드 성공")
except FileNotFoundError:
    print(f"지정된 CSV 파일이 존재하지 않습니다: {csv_file_path}")
    exit()

# 경로 수정 (절대 경로로 변경)
df['frame_path'] = df['frame_path'].apply(lambda x: os.path.join(base_path, x))

# 존재하지 않는 파일 확인
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

# 결과 출력
print(f"최종 누락된 파일 개수: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"누락된 파일 예시: {missing_files[:5]}")
else:
    print("모든 파일이 존재합니다.")
