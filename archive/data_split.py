import os
import shutil
from sklearn.model_selection import train_test_split

# ==== 경로 설정 ====
data_dir = r"C:\Users\USER\jetson_nano_dataset"   # 원본 이미지 경로 (5개 폴더)
train_dir = r"C:\Users\USER\train_image"   # 학습 데이터 저장 경로
val_dir = r"C:\Users\USER\val_image"       # 검증 데이터 저장 경로
test_dir = r"C:\Users\USER\test_image"     # 테스트 데이터 저장 경로

# ==== 이미지 나누기 및 저장 ====
def split_and_save_images(folder_path, folder_name):
    # 이미지 파일 수집
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    print(f"{folder_name} 이미지 수: {len(images)}")
    
    # 데이터 나누기 (7:2:1 비율)
    train_images, remaining_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(remaining_images, test_size=0.33, random_state=42)
    
    # 이미지 복사 함수
    def copy_images(image_list, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for image in image_list:
            shutil.copy(os.path.join(folder_path, image), os.path.join(target_dir, image))
    
    # 각 폴더에 이미지 저장
    copy_images(train_images, os.path.join(train_dir, folder_name))
    copy_images(val_images, os.path.join(val_dir, folder_name))
    copy_images(test_images, os.path.join(test_dir, folder_name))
    
    print(f"{folder_name} -> Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

# ==== 각 폴더에 대해 작업 ====
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):  # 폴더인지 확인
        split_and_save_images(folder_path, folder_name)

print("모든 폴더에 대해 데이터셋 분할 및 저장이 완료되었습니다.")
