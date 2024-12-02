import cv2
import numpy as np
import os
import csv

# 데이터 저장 경로
output_dir = "C:/Users/USER/finish_line"
os.makedirs(output_dir, exist_ok=True)

# CSV 파일 생성
csv_file = os.path.join(output_dir, "labels.csv")
csv_data = [["image_path", "center_x", "center_y"]]

# 이미지 리스트 (데이터가 저장된 경로에서 읽기)
image_folder = "C:/Users/USER/line"  # 원본 이미지 경로
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 선분 기울기 계산 함수
def calculate_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:  # 수직선인 경우
        return float('inf')
    return (y2 - y1) / (x2 - x1)

# 이미지 처리 및 라벨 생성
for i, image_file in enumerate(image_files):
    # 원본 이미지 로드
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Failed to load image at '{image_path}'. Skipping this file.")
        continue

    # 그레이스케일 변환 및 가우시안 블러
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지 검출
    edges = cv2.Canny(blur, 50, 150)

    # 허프 변환으로 라인 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # 빈 캔버스 생성
    line_image = np.zeros_like(image)

    # 라인 중심 좌표 계산 변수
    center_x, center_y = -1, -1

    # 라인 연결을 위한 리스트
    lines_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines_list.append((x1, y1, x2, y2))

        # 선분 연결을 위한 정렬 및 기울기 기반 연결
        for i in range(1, len(lines_list)):
            x1, y1, x2, y2 = lines_list[i-1]
            x3, y3, x4, y4 = lines_list[i]

            # 선들의 기울기 비교 (기울기가 비슷하면 연결 가능성 있음)
            slope1 = calculate_slope(x1, y1, x2, y2)
            slope2 = calculate_slope(x3, y3, x4, y4)

            # 기울기가 유사하고, 끝점 간의 거리가 가까운 경우 선 연결
            if abs(slope1 - slope2) < 0.1:  # 기울기 차이 기준 (조정 가능)
                if np.sqrt((x2 - x3)**2 + (y2 - y3)**2) < 10:  # 끝점 간 거리 기준
                    cv2.line(line_image, (x1, y1), (x4, y4), (255, 255, 255), 2)
                else:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            else:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # 라인의 중심 계산
        x_coords = []
        y_coords = []
        for line in lines_list:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

        if x_coords and y_coords:
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))

    # 이미지 크기 키우기 (원래 크기의 2배로 확장)
    enlarged_image = cv2.resize(line_image, (line_image.shape[1] * 2, line_image.shape[0] * 2))

    # 처리된 이미지 저장
    processed_image_path = os.path.join(output_dir, f"processed_{i}.jpg")
    cv2.imwrite(processed_image_path, enlarged_image)

    # 라벨 데이터 저장
    csv_data.append([processed_image_path, center_x, center_y])

# CSV 파일 저장
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print("데이터 처리 및 저장 완료!")
