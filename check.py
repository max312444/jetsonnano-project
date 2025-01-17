import os
import shutil
import pandas as pd

# **1. 경로 설정**
old_base_path = r"C:\Users\USER\dataset_0117"
new_base_path = r"C:\Users\USER\augmented_frames"
csv_path = r"C:\Users\USER\data.csv"
updated_csv_path = r"C:\Users\USER\training_data_cleaned_updated.csv"

# **2. 기존 디렉토리 및 파일 확인**
if not os.path.exists(old_base_path):
    print(f"경로가 존재하지 않습니다: {old_base_path}")
else:
    files = os.listdir(old_base_path)
    if not files:
        print(f"{old_base_path} 경로에 파일이 없습니다.")
    else:
        print(f"총 {len(files)}개의 파일이 발견되었습니다.")
        print(f"파일 예시: {files[:5]}")

# **3. 새로운 디렉토리 생성**
os.makedirs(new_base_path, exist_ok=True)

# **4. 파일 복사 및 이름 변경**
for file_name in os.listdir(old_base_path):
    old_file_path = os.path.join(old_base_path, file_name)
    new_file_path = os.path.join(new_base_path, f"cropped_{file_name}")
    if os.path.isfile(old_file_path):  # 파일만 복사
        try:
            shutil.copy(old_file_path, new_file_path)
        except Exception as e:
            print(f"파일 복사 중 에러 발생: {old_file_path}, 에러: {e}")

# **5. 파일 복사 결과 확인**
all_files = os.listdir(new_base_path)
cropped_files = [file for file in all_files if file.startswith("cropped_")]

print(f"총 {len(cropped_files)}개의 'cropped_' 파일이 복사되었습니다.")
print(f"복사된 파일 예시: {cropped_files[:5]}")

# **6. CSV 파일 로드 및 경로 수정**
df = pd.read_csv(csv_path)
df['frame_path'] = df['frame_path'].apply(
    lambda x: os.path.join(new_base_path, f"cropped_{os.path.basename(x)}")
)

# **7. 수정된 CSV 저장**
df.to_csv(updated_csv_path, index=False)
print(f"수정된 CSV 파일이 저장되었습니다: {updated_csv_path}")

# **8. 누락된 파일 확인**
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

print(f"누락된 파일 개수: {len(missing_files)}")
if missing_files:
    print(f"누락된 파일 예시: {missing_files[:5]}")
else:
    print("모든 파일이 정상적으로 존재합니다.")
