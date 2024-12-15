from PIL import Image
import os

# 원본 이미지 폴더
input_folder = r"C:\Users\USER\jetson_nano_dataset"

# 폴더 내 모든 파일 처리
files = os.listdir(input_folder)

for filename in files:
    image_path = os.path.join(input_folder, filename)
    
    # 이미지 파일 확인
    try:
        # 이미지 열기
        img = Image.open(image_path)
        img.show()  # 이미지 보기
        print(f"이미지 열기 성공: {image_path}")
    except Exception as e:
        print(f"이미지 열기 실패: {image_path}, 오류: {e}")
