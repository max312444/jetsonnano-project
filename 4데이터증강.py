from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 이미지 폴더 경로
image_folder_path = "C:/Users/USER/dataset_0117"

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# 첫 번째 이미지 불러오기
image_path = os.path.join(image_folder_path, image_files[0])
image = cv2.imread(image_path)

# 이미지 증강을 위한 ImageDataGenerator 설정
datagen = ImageDataGenerator(
    rotation_range=40,         # 0~40도 회전
    width_shift_range=0.2,     # 수평 이동
    height_shift_range=0.2,    # 수직 이동
    shear_range=0.2,           # 전단 변환
    zoom_range=0.2,            # 확대/축소
    horizontal_flip=True,      # 좌우 반전
    fill_mode='nearest'        # 증강 후 비어있는 부분 채우기 방법
)

# 이미지를 정규화하여 (0~1 범위로) 변환
image_normalized = image.astype(np.float32) / 255.0
image_normalized = np.expand_dims(image_normalized, axis=0)  # 배치 차원 추가

# 증강된 이미지를 생성
i = 0
for batch in datagen.flow(image_normalized, batch_size=1, save_to_dir='C:/Users/USER/dataset_0117/augmented_images', save_prefix='aug', save_format='jpeg'):
    i += 1
    if i > 5:  # 5개의 증강 이미지만 생성
        break

# 증강된 이미지 중 하나를 확인
augmented_image_path = 'C:/Users/USER/dataset_0117/augmented_images/aug_0_0.jpeg'
augmented_image = cv2.imread(augmented_image_path)

# 증강된 이미지 출력 (RGB로 변환)
augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
plt.imshow(augmented_image_rgb)
plt.title("증강된 이미지")
plt.axis('off')
plt.show()
