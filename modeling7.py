import os
import cv2
from PIL import Image
import numpy as np

# 원본 이미지 폴더
input_folder = r"C:\Users\USER\jetson_nano_dataset"

# 증강 이미지 저장 폴더
output_folder = r"C:\Users\USER\finish_image"
os.makedirs(output_folder, exist_ok=True)

# 모델에 맞는 크기 (200x66)
target_size = (200, 66)

# 이미지 유효성 검사 함수
def is_valid_image(file_path):
    try:
        # OpenCV로 파일 열기 시도
        image = cv2.imread(file_path)
        if image is None:
            # Pillow로 열기 시도
            with Image.open(file_path) as img:
                img.verify()  # 이미지가 유효한지 확인
        return True  # 이미지 파일로 확인됨
    except Exception:
        return False  # 유효하지 않은 파일

# 폴더 내 모든 파일을 탐색
valid_files = []
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        # 숨김 파일 제외하고, 이미지 유효성 검사
        file_path = os.path.join(root, filename)
        if not filename.startswith('.') and is_valid_image(file_path):
            valid_files.append(file_path)

print(f"유효한 이미지 파일들: {valid_files}")  # 유효한 이미지 파일 출력

# 이미지 전처리
for image_path in valid_files:
    filename = os.path.basename(image_path)
    print(f"\n파일 확인 중: {filename}")

    try:
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            with Image.open(image_path).convert("RGB") as img:
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 크기 조정
        resized_image = cv2.resize(image, target_size)
        output_filename = f"cropped_{os.path.splitext(filename)[0]}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # 이미지 저장
        cv2.imwrite(output_path, resized_image)
        print(f"전처리된 이미지를 저장했습니다: {output_path}")

    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {image_path}, 오류 내용: {e}")

print("모든 이미지의 전처리가 완료되었습니다.")
