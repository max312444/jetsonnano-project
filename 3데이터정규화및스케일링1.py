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

# 이미지 정규화 (픽셀 값을 0~1로 변환)
image_normalized = image.astype(np.float32) / 255.0

# 정규화된 이미지 출력 (OpenCV는 BGR 형식이라 RGB로 변환)
image_rgb = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB)

# 이미지 확인
plt.imshow(image_rgb)
plt.title(f"정규화된 이미지: {image_files[0]}")
plt.axis('off')
plt.show()

# 정규화된 이미지 크기 출력
print(f"정규화된 이미지 크기: {image_normalized.shape}")
