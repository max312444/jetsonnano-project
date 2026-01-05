import cv2
import numpy as np

# 이미지 파일 경로 생성 (훈련 데이터 예시)
train_images = [os.path.join("C:/Users/USER/dataset_0117", img_name) for img_name in X_train]

# 이미지를 훈련 데이터에 맞게 불러오기
train_image_data = []
for img_path in train_images:
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (224, 224))  # 이미지 크기 조정 (예시로 224x224로 설정)
    train_image_data.append(image_resized)

# numpy 배열로 변환
train_image_data = np.array(train_image_data)

# 훈련 데이터 이미지 크기 확인
print(f"훈련 데이터 이미지 크기: {train_image_data.shape}")

# 같은 방식으로 테스트 데이터도 처리할 수 있습니다.
