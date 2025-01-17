import os
import pandas as pd
import cv2
import shutil

# **1. CSV 파일 경로**
csv_path = r"C:\Users\USER\data.csv"

# **2. 데이터 로드**
df = pd.read_csv(csv_path)

# **3. 경로 설정**
old_base_path = r"C:\Users\USER\dataset_0117"
new_base_path = r"C:\Users\USER\augmented_frames"
os.makedirs(new_base_path, exist_ok=True)

# 기존 경로를 새로운 경로로 변경 (파일명에 "cropped_" 추가)
df['frame_path'] = df['frame_path'].apply(
    lambda x: os.path.join(new_base_path, f"cropped_{os.path.basename(x)}")
)

# **4. 누락된 파일 처리**
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

for path in missing_files:
    original_path = os.path.join(old_base_path, os.path.basename(path).replace("cropped_", ""))
    if os.path.exists(original_path):
        # 전처리하여 저장
        image = cv2.imread(original_path)
        if image is not None:
            resized_image = cv2.resize(image, (200, 66))  # 크기 조정
            cv2.imwrite(path, resized_image)
            print(f"전처리 후 저장: {path}")
        else:
            print(f"이미지 로드 실패: {original_path}")
    else:
        print(f"원본 파일 없음: {original_path}")

# **5. 누락된 파일 제거**
existing_files = [path for path in df['frame_path'] if os.path.exists(path)]
df_cleaned = df[df['frame_path'].isin(existing_files)]

# **6. 결과 저장**
updated_csv_path = r"C:\Users\USER\training_data_cleaned_updated.csv"
df_cleaned.to_csv(updated_csv_path, index=False)

# **7. 결과 출력**
print(f"총 누락된 파일 개수: {len(missing_files)}")
print(f"최종 데이터셋 크기: {len(df_cleaned)}")
print(f"경로가 수정된 CSV를 {updated_csv_path}에 저장했습니다.")
