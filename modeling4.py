import cv2
import os
import csv

# 데이터 경로 설정
base_path = r"C:\Users\USER\jetson_csv\steering_data.csv"
frame_save_path = os.path.join(base_path, r"C:\Users\USER\jetson_nano_dataset")
csv_file = os.path.join(base_path, r"C:\Users\USER\jetson_csv\steering_data.csv")

# CSV 파일 로드
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# 헤더 제거
data = data[1:]

# 인덱스 초기화
current_index = 0

def load_image_and_angle(index):
    """주어진 인덱스의 이미지와 각도를 로드"""
    frame_path, angle = data[index]
    
    # 각도 값 출력
    print(f"Loaded image: {frame_path}, Angle: {angle}")  # 각도 출력
    
    # 각도 값이 문자인지 확인하고 float로 변환
    angle = angle.strip()  # 공백 제거
    try:
        angle = float(angle)  # 각도 값을 float로 변환
    except ValueError:
        print(f"각도 변환 오류: {angle}를 float로 변환할 수 없습니다.")
        return None, None
    
    image = cv2.imread(frame_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {frame_path}")
        return None, None
    return image, angle

def delete_current_data(index):
    """현재 인덱스의 데이터를 삭제"""
    frame_path, _ = data[index]
    if os.path.exists(frame_path):
        os.remove(frame_path)
    del data[index]
    print(f"Deleted: {frame_path}")
    # CSV 파일 갱신
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_path', 'steering_angle'])
        writer.writerows(data)

# 윈도우 설정
cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)

while True:
    # 현재 이미지와 각도 로드
    if current_index < 0 or current_index >= len(data):
        print("더 이상 데이터가 없습니다.")
        break

    image, angle = load_image_and_angle(current_index)
    if image is None:
        break

    # 이미지와 각도 출력sd
    display_image = image.copy()
    cv2.putText(display_image, f"Angle: {angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image Viewer", display_image)

    # 키 입력 처리
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):  # 'q'를 누르면 종료
        break
    elif key == ord('a'):  # 'a'를 누르면 이전 이미지
        current_index = max(0, current_index - 1)
    elif key == ord('d'):  # 'd'를 누르면 다음 이미지
        current_index = min(len(data) - 1, current_index + 1)
    elif key == ord('c'):  # 'c'를 누르면 현재 이미지 삭제
        delete_current_data(current_index)
        current_index = max(0, current_index - 1)

cv2.destroyAllWindows()
