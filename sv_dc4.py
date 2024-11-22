import os
import Jetson.GPIO as GPIO
import time
import cv2
import keyboard

# 폴더 생성 (출력 파일 저장 경로)
output_dir = "./jwj"
output_file = os.path.join(output_dir, "output.avi")

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

# 서보 모터 각도 초기화
angle = 90

# DC 모터 상태 초기화
dc_motor_speed = 0
dc_motor_direction = "forward"

# 녹화 상태 초기화
recording = False
out = None

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'H264')

# 서보 모터 각도 설정 함수
def set_servo_angle(new_angle):
    global angle
    angle = max(0, min(180, new_angle))  # 각도 범위 제한
    duty_cycle = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    servo.ChangeDutyCycle(0)
    print(f"Servo angle set to {angle} degrees")

# DC 모터 방향과 속도 조절 함수
def set_dc_motor(speed, direction):
    global dc_motor_speed, dc_motor_direction
    dc_motor_speed = speed
    dc_motor_direction = direction
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)
    print(f"DC motor moving {direction} at speed {speed}")

# 녹화 토글 함수
def toggle_recording():
    global recording, out
    if recording:
        recording = False
        if out is not None:
            out.release()
        print("Recording stopped.")
    else:
        print(f"Recording to: {output_file}")
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (1280, 720))
        recording = True
        print("Recording started.")

# 키보드 핫키 설정
keyboard.add_hotkey('r', toggle_recording)
keyboard.add_hotkey('0', lambda: set_servo_angle(90))
keyboard.add_hotkey('left', lambda: set_servo_angle(angle - 5))
keyboard.add_hotkey('right', lambda: set_servo_angle(angle + 5))
keyboard.add_hotkey('up', lambda: set_dc_motor(50, "forward"))
keyboard.add_hotkey('down', lambda: set_dc_motor(50, "backward"))

# 메인 루프
try:
    print("Use arrow keys to control. Press 'r' to start/stop recording.")
    print("Press '0' to reset servo angle to 90 degrees. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 화면에 프레임 출력
        cv2.imshow('Webcam Feed', frame)

        # 녹화 중이면 프레임 저장
        if recording and out is not None:
            out.write(frame)

        # 'q' 키로 종료
        if keyboard.is_pressed('q'):
            print("Exiting...")
            break

        # CPU 사용량 줄이기
        time.sleep(0.05)

finally:
    if recording and out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("Clean up complete.")
