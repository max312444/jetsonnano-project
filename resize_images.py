import os
import cv2

def resize_images(input_path, output_path, target_size=(66, 200)):
    # input_path: 원본 이미지가 저장된 디렉토리
    # output_path: 리사이즈된 이미지가 저장될 디렉토리
    os.makedirs(output_path, exist_ok=True)
    
    # jetson_nano_dataset 안의 모든 폴더 탐색
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        if os.path.isdir(folder_path):  # 폴더인 경우만 탐색
            # 각 폴더 안에 있는 하위 폴더를 탐색
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):  # 하위 폴더가 존재할 때
                    # 하위 폴더 안에 있는 이미지 파일을 탐색
                    for file in os.listdir(subfolder_path):
                        if file.endswith(('.jpg', '.png')):  # 이미지 파일만 선택
                            img_path = os.path.join(subfolder_path, file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                resized_img = cv2.resize(img, target_size)
                                output_subfolder = os.path.join(output_path, folder, subfolder)
                                os.makedirs(output_subfolder, exist_ok=True)
                                cv2.imwrite(os.path.join(output_subfolder, file), resized_img)

# 원본 이미지 경로와 리사이즈된 이미지 저장 경로 설정
input_path = r"C:\Users\USER\jetson_nano_dataset"   # 원본 이미지 폴더 경로 (새 경로로 수정)
output_path = r"C:\Users\USER\change_image"         # 리사이즈된 이미지 저장 경로 (새 경로로 수정)

# 리사이징 함수 호출
resize_images(input_path, output_path)
