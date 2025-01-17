import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터 로드 및 탐색
csv_file_path = "C:/Users/USER/dataset_0117/data.csv"
df = pd.read_csv(csv_file_path)

# 이미지 파일명과 레이블을 분리
image_names = df['image_name'].values
angles = df['angle'].values
speeds = df['speed'].values

# 2. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(image_names, 
                                                    pd.DataFrame({'angle': angles, 'speed': speeds}),
                                                    test_size=0.2, random_state=42)

# 3. 이미지 로딩 및 정규화
image_folder_path = "C:/Users/USER/dataset_0117"
train_images = [os.path.join(image_folder_path, img_name) for img_name in X_train]
test_images = [os.path.join(image_folder_path, img_name) for img_name in X_test]

# 이미지를 로드하고 크기를 224x224로 조정
train_image_data = []
for img_path in train_images:
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (224, 224))  # 이미지 크기 조정
    image_normalized = image_resized.astype(np.float32) / 255.0  # 정규화
    train_image_data.append(image_normalized)

test_image_data = []
for img_path in test_images:
    image = cv2.imread(img_path)
    image_resized = cv2.resize(image, (224, 224))  # 이미지 크기 조정
    image_normalized = image_resized.astype(np.float32) / 255.0  # 정규화
    test_image_data.append(image_normalized)

# numpy 배열로 변환
train_image_data = np.array(train_image_data)
test_image_data = np.array(test_image_data)

# 4. 데이터 증강 (ImageDataGenerator)
datagen = ImageDataGenerator(
    rotation_range=40,         
    width_shift_range=0.2,     
    height_shift_range=0.2,    
    shear_range=0.2,           
    zoom_range=0.2,            
    horizontal_flip=True,      
    fill_mode='nearest'
)

# 5. CNN 모델 정의
model = Sequential()

# 첫 번째 합성곱 층
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

# 두 번째 합성곱 층
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 세 번째 합성곱 층
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten 층
model.add(Flatten())

# 완전 연결 층
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# 출력 층 (각도와 속도 예측)
model.add(Dense(2))  # 두 개의 출력을 가진 모델 (각도, 속도)

# 모델 컴파일
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# 모델 요약 출력
model.summary()

# 6. 모델 학습
history = model.fit(train_image_data, y_train.values, 
                    epochs=10, batch_size=32, 
                    validation_data=(test_image_data, y_test.values))

# 7. 모델 성능 평가
test_loss, test_mae = model.evaluate(test_image_data, y_test.values)
print(f"테스트 손실: {test_loss}")
print(f"테스트 MAE (평균 절대 오차): {test_mae}")

# 8. 학습 과정 시각화
# 손실 함수 그래프
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.title('손실 함수')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 평균 절대 오차 (MAE) 그래프
plt.plot(history.history['mae'], label='훈련 MAE')
plt.plot(history.history['val_mae'], label='검증 MAE')
plt.title('평균 절대 오차 (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# 9. 모델 저장
model.save('autonomous_car_model.h5')

# 10. 모델 불러오기 (나중에 추론 시 사용)
# from keras.models import load_model
# model = load_model('autonomous_car_model.h5')
