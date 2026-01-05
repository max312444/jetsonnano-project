import os
import Jetson.GPIO as GPIO
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import threading
from datetime import datetime

# 모델 경로
MODEL_PATH = r"C:\Users\USER\best_pilotnet_model.pth"

# 폴더 생성 (출력 파일 저장 경로)
output_dir = "./jwj"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# GPIO 핀 설정
servo_pin = 33
dc_motor_pwm_pin = 32  # DC 모터 속도 제어 핀
dc_motor_dir_pin1 = 29  # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31  # DC 모터 방향 제어 핀 2

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
servo.start(0)
dc_motor_pwm.start(0)

# 서보 모터 각도 설정 함수
def set_servo_angle(angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)

# DC 모터 방향과 속도 조절 함수
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 모델 초기화
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.model(x)

# 모델 로드
def load_model(model_path):
    model = PilotNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model(MODEL_PATH)

# 전처리 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 카메라 처리 쓰레드
class CameraHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 모델로 각도 예측
            input_frame = transform(frame).unsqueeze(0)
            with torch.no_grad():
                angle = model(input_frame).item()

            # 예측된 각도로 서보 모터 제어
            servo_angle = int((angle + 1) * 90)  # 모델 출력 범위가 -1~1일 경우
            set_servo_angle(servo_angle)

            # DC 모터 제어 (예제: 항상 앞으로 이동)
            set_dc_motor(50, "forward")

            # 화면 출력
            cv2.putText(frame, f"Angle: {servo_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Autonomous Car', frame)

            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 메인 함수
def main():
    camera_handler = CameraHandler()

    try:
        camera_handler.start()

        print("Press 'q' to exit.")
        while camera_handler.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("프로그램 종료 중...")

    finally:
        camera_handler.running = False
        camera_handler.join()
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()
        print("정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
