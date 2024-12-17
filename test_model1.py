import torch
import torchvision.transforms as transforms
import cv2
from datetime import datetime

# 모델 로드
model_path = r"C:\Users\USER\steering_model.pth"  # 훈련된 모델 파일 경로
model = torch.load(model_path)
model.eval()  # 평가 모드로 설정

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # 모델 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 테스트 함수
def test_model(model):
    cap = cv2.VideoCapture(0)  # 카메라 캡처 시작
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 이미지 전처리 및 예측
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0)  # 배치 차원 추가
        with torch.no_grad():
            prediction = model(image)

        # 예측 결과 출력
        _, predicted = torch.max(prediction, 1)
        print(f"Predicted: {predicted.item()}")  # 예측된 클래스 출력

        # 예측 결과를 바탕으로 실제 모델이 수행해야 할 작업 (예: 서보 각도, 전진/후진 등)
        # 여기에 서보 제어나 모터 제어 코드를 추가할 수 있습니다.
        # 예를 들어:
        # if predicted.item() == 0:  # 직진
        #     set_dc_motor(80, "forward")
        # elif predicted.item() == 1:  # 왼쪽
        #     set_servo_angle(70)
        # elif predicted.item() == 2:  # 오른쪽
        #     set_servo_angle(110)

        # 화면에 실시간 영상 출력
        cv2.imshow('Test Webcam Feed', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 함수
if __name__ == "__main__":
    test_model(model)
