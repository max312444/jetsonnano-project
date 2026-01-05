import torch
from model import DrivingModel
from PIL import Image
import numpy as np

# 모델 불러오기
model = DrivingModel()
model.load_state_dict(torch.load(r'C:\Users\USER\jetson_nano_dataset\driving_model.pth'))  # 모델 파일 경로
model.eval()

# 테스트 이미지 로딩
test_image_path = r'C:\Users\USER\change_image\test_image.jpg'  # 테스트할 이미지 경로
image = Image.open(test_image_path)
image = np.array(image)

# 이미지 전처리 (모델에 맞는 크기 및 차원 변경 필요)
image = torch.tensor(image).float().unsqueeze(0)  # 배치 차원 추가

# 모델 예측
with torch.no_grad():
    steering_angle = model(image)
    print(f'Predicted Steering Angle: {steering_angle.item()}')
