import os
from PIL import Image

# 이미지 폴더 경로
input_folder = r"C:\Users\USER\jetson_nano_dataset"

# 폴더 내 파일 목록
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # 파일인지 확인 (폴더 제외)
    if not os.path.isfile(file_path):
        print(f"폴더이므로 건너뜀: {filename}")
        continue
    
    try:
        # 이미지 무결성 검사
        with Image.open(file_path) as img:
            img.verify()
        print(f"정상 이미지 파일: {filename}")
    except Exception as e:
        print(f"손상된 파일 또는 접근 불가: {filename}, 오류: {e}")

print("이미지 무결성 검사 완료.")
