import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_file_path = "C:/Users/USER/dataset_0117/data.csv"

# CSV 파일 로드
df = pd.read_csv(csv_file_path)

# CSV 파일에서 첫 번째 레이블과 이미지 파일명 확인
image_name = df.iloc[0]['image_name']  # 'image_name'은 CSV의 컬럼명 (파일명)
angle = df.iloc[0]['angle']  # 'angle'은 각도 레이블 (예시)
speed = df.iloc[0]['speed']  # 'speed'는 속도 레이블 (예시)

# 이미지 경로
image_folder_path = "C:/Users/USER/dataset_0117"
image_path = os.path.join(image_folder_path, image_name)

# 이미지 불러오기
image = cv2.imread(image_path)

# 이미지 출력 (BGR -> RGB 변환)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title(f"이미지: {image_name}\n각도: {angle}°\n속도: {speed} km/h")
plt.axis('off')
plt.show()

# 레이블 확인
print(f"이미지 파일명: {image_name}")
print(f"각도: {angle}°")
print(f"속도: {speed} km/h")
