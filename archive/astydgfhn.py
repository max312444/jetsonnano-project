import os
import pandas as pd
import cv2

# **1. CSV 파일 경로**
csv_path = r"C:\Users\USER\data.csv"

# **2. 데이터 로드**
df = pd.read_csv(csv_path)

# **3. 경로 설정**
old_base_path = r"C:\Users\USER\dataset_0117"
new_base_path = r"C:\Users\USER\augmented_frames"
os.makedirs(new_base_path, exist_ok=True)

# **4. 경로 업데이트 (디버깅 추가)**
def update_path(x):
    new_path = os.path.join(new_base_path, f"cropped_{os.path.basename(x)}")
    print(f"Updated path: {new_path}")  # 디버깅용 출력
    return new_path

df['frame_path'] = df['frame_path'].apply(update_path)

# **5. 누락된 파일 처리 (디버깅 추가)**
missing_files = [path for path in df['frame_path'] if not os.path.exists(path)]

for path in missing_files:
    original_path = os.path.join(old_base_path, os.path.basename(path).replace("cropped_", ""))
    print(f"Checking original path: {original_path}")
    
    if os.path.exists(original_path):
        # 전처리 후 저장
        image = cv2.imread(original_path)
        if image is not None:
            resized_image = cv2.resize(image, (200, 66))  # 크기 조정
            cv2.imwrite(path, resized_image)
            print(f"Processed and saved: {path}")
        else:
            print(f"Image load failed: {original_path}")
    else:
        print(f"Original file not found: {original_path}")

# **6. 누락된 파일 제거**
existing_files = [path for path in df['frame_path'] if os.path.exists(path)]
df_cleaned = df[df['frame_path'].isin(existing_files)]

# **7. 결과 저장**
updated_csv_path = r"C:\Users\USER\training_data_cleaned_updated.csv"
df_cleaned.to_csv(updated_csv_path, index=False)

# **8. 결과 출력**
print(f"총 누락된 파일 개수: {len(missing_files)}")
print(f"최종 데이터셋 크기: {len(df_cleaned)}")
print(f"경로가 수정된 CSV를 {updated_csv_path}에 저장했습니다.")
