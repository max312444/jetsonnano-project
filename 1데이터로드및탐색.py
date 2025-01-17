import cv2
import os
import matplotlib.pyplot as plt

# 이미지 폴더 경로
image_folder_path = "C:/Users/USER/dataset_0117"

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# 첫 번째 이미지 불러오기
image_path = os.path.join(image_folder_path, image_files[0])
image = cv2.imread(image_path)

# 이미지 크기 출력
print(f"이미지 크기: {image.shape}")  # (height, width, channels)

# 이미지 출력 (OpenCV는 BGR 형식이라 RGB로 변환)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 확인
plt.imshow(image_rgb)
plt.title(f"첫 번째 이미지: {image_files[0]}")
plt.axis('off')
plt.show()
