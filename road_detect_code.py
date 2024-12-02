import os
import Jetson.GPIO as GPIO
import time
import cv2
import keyboard
import threading
from datetime import datetime
import subprocess
import numpy as np  # NumPy 추가

# GPIO 핀에서 PWM 기능을 활성화하는 함수
def enable_pwm_on_pins():
    subprocess.run(["sudo", "busybox", "devmem", "0x700031fc", "32", "0x45"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d504", "32", "0x2"])
    subprocess.run(["sudo", "busybox", "devmem", "0x70003248", "32", "0x46"])
    subprocess.run(["sudo", "busybox", "devmem", "0x6000d100", "32", "0x00"])

# 폴더 생성
output_dir = "./jwj"
folders = {45: os.path.join(output_dir, "45"), 
           90: os.path.join(output_dir, "90"), 
           135: os.path.join(output_dir, "135")}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

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
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)

# DC 모터 방향과 속도 조절 함수
def set_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 각도에 따라 이미지를 저장할 폴더 선택
def save_image(frame, angle):
    nearest_angle = min(folders.keys(), key=lambda x: abs(x - angle))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filepath = os.path.join(folders[nearest_angle], f"image_{timestamp}.jpg")
    cv2.imwrite(filepath, frame)
    print(f"Image saved at: {filepath}")

# 도로 영역만 인식하는 함수 (색상 필터링)
def detect_road(frame):
    # BGR -> HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 흰색 도로 색상 범위 설정 (HSV)
    lower_road_color = np.array([0, 0, 100])  # 하얀 도로 영역 (예시)
    upper_road_color = np.array([180, 255, 255])

    # 색상 범위에 해당하는 부분을 마스크로 추출
    mask = cv2.inRange(hsv, lower_road_color, upper_road_color)

    # 마스크를 이용해 원본 이미지에서 도로 영역만 추출
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result

# 카메라 처리 쓰레드
class CameraHandler(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.current_angle = 90
        self.should_save_image = False

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 도로 영역만 추출 (색상 필터링)
            road_frame = detect_road(frame)

            cv2.imshow('Webcam Feed', road_frame)

            # 전진 중일 때 이미지를 저장
            if self.should_save_image:
                save_image(road_frame, self.current_angle)
                self.should_save_image = False  # 한 번 저장한 후 대기 상태로

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

# 모터 제어 쓰레드
class MotorHandler(threading.Thread):
    def __init__(self, camera_handler):
        super().__init__()
        self.camera_handler = camera_handler
        self.servo_angle = 90
        set_servo_angle(self.servo_angle)
        self.running = True

    def run(self):
        while self.running:
            if keyboard.is_pressed('0'):
                self.servo_angle = 90
                set_servo_angle(self.servo_angle)
                print("Servo angle reset to 90 degrees.")

            elif keyboard.is_pressed('left'):
                self.servo_angle -= 5
                if self.servo_angle < 0:
                    self.servo_angle = 0
                set_servo_angle(self.servo_angle)
                print(f"Left pressed. Servo angle: {self.servo_angle} degrees")

            elif keyboard.is_pressed('right'):
                self.servo_angle += 5
                if self.servo_angle > 180:
                    self.servo_angle = 180
                set_servo_angle(self.servo_angle)
                print(f"Right pressed. Servo angle: {self.servo_angle} degrees")

            # Update current angle in CameraHandler
            self.camera_handler.current_angle = self.servo_angle

            if keyboard.is_pressed('up'):
                set_dc_motor(50, "forward")
                print("DC motor moving forward...")
                self.camera_handler.should_save_image = True  # 전진 시 이미지 저장 플래그 설정
            elif keyboard.is_pressed('down'):
                set_dc_motor(50, "backward")
                print("DC motor moving backward...")
            else:
                set_dc_motor(0, "forward")

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
