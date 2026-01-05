import os
import Jetson.GPIO as GPIO
import time
import cv2
import threading
from datetime import datetime
import subprocess
import torch
import torchvision.transforms as transforms

# 모델 로드
model_path = r"C:\Users\USER\steering_model.pth"  # 훈련된 모델 파일 경로
model = torch.load(model_path)
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # 모델 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# GPIO 핀에서 PWM 기능을 활성화하는 함수
def enable_pwm_on_pins():
    subprocess.run(["sudo", "busybox", "devmem", "0x700031fc", "32", "0x45"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d504", "32", "0x2"])
    subprocess.run(["sudo", "busybox", "devmem", "0x70003248", "32", "0x46"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d100", "32", "0x00"])

# GPIO 핀 설정
servo_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

# PWM 활성화
enable_pwm_on_pins()

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
    servo.ChangeDutyCycle(0)  # 대기 시간 제거

# DC 모터 방향과 속도 조절 함수
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 카메라 처리 쓰레드
class CameraHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.current_angle = 90

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            cv2.imshow('Webcam Feed', frame)

            # 이미지 전처리 및 예측
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0)  # 배치 차원 추가
            with torch.no_grad():
                prediction = model(image)
            
            # 모델의 출력값을 기반으로 서보 각도 계산 (예시)
            servo_angle = self.get_servo_angle_from_prediction(prediction)
            set_servo_angle(servo_angle)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def get_servo_angle_from_prediction(self, prediction):
        # 모델의 출력에 따라 서보 각도를 결정하는 예시 함수
        # 이 부분은 모델이 어떻게 훈련되었는지에 따라 다르게 작성해야 함
        _, predicted = torch.max(prediction, 1)
        if predicted.item() == 0:  # 예: 0은 직진
            return 90
        elif predicted.item() == 1:  # 예: 1은 왼쪽
            return 70
        elif predicted.item() == 2:  # 예: 2는 오른쪽
            return 110
        return 90

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 모터 제어 쓰레드
class MotorHandler(threading.Thread):
    def __init__(self, camera_handler):
        super().__init__()
        self.camera_handler = camera_handler
        self.servo_angle = 90  # 초기 각도는 90도
        set_servo_angle(self.servo_angle)
        self.running = True

    def run(self):
        while self.running:
            # 모델 예측에 따른 서보 각도 업데이트
            self.camera_handler.current_angle = self.servo_angle

            # 전진 제어
            set_dc_motor(80, "forward")
            time.sleep(0.1)

    def stop(self):
        self.running = False

# 메인 함수
def main():
    camera_handler = CameraHandler()
    motor_handler = MotorHandler(camera_handler)

    try:
        camera_handler.start()
        motor_handler.start()

        print("Press 'q' to exit.")
        while camera_handler.running:
            pass

    except KeyboardInterrupt:
        print("프로그램 종료 중...")

    finally:
        camera_handler.running = False
        motor_handler.stop()
        camera_handler.join()
        motor_handler.join()
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()
        print("정상적으로 종료되었습니다.")

if __name__ == "__main__":
    main()
