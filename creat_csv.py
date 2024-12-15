import os
import pandas as pd

# 이미지 디렉토리 및 CSV 저장 경로 설정
image_dir = r"C:\Users\USER\jetson_nano_dataset"
csv_file = r"C:\Users\USER\jetson_csv\jetson_data.csv"  # 파일명 포함

def create_csv(image_dir, csv_file):
    """
    이미지 디렉토리 내의 모든 이미지를 찾아 'filename'과 'steering_angle'을 포함하는 CSV 파일 생성.
    하위 폴더까지 순회하여 이미지를 처리.
    """
    # 디렉토리 확인 및 생성
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))  # 디렉토리가 없으면 생성

    data = []

    # 이미지 폴더 내의 모든 파일을 확인 (하위 폴더까지 순회)
    for root, dirs, files in os.walk(image_dir):
        for image_name in files:
            if image_name.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일만 선택
                # 이미지 경로 생성
                image_path = os.path.join(root, image_name)

                # 수동으로 steering_angle 값을 설정
                # 예시: 이미지 이름에 따른 조향각도 설정
                if "left" in image_name:
                    steering_angle = -0.1  # 왼쪽으로 돌린 이미지는 -0.1
                elif "right" in image_name:
                    steering_angle = 0.1  # 오른쪽으로 돌린 이미지는 0.1
                else:
                    steering_angle = 0.0  # 기본적으로 중앙은 0.0

                # 데이터를 리스트에 추가
                print(f"파일명: {image_name}, 경로: {image_path}, 조향각도: {steering_angle}")  # 디버깅 출력
                data.append([image_path, steering_angle])

    # DataFrame으로 변환하여 CSV 파일로 저장
    if data:
        df = pd.DataFrame(data, columns=['filename', 'steering_angle'])
        df.to_csv(csv_file, index=False)
        print(f"CSV 파일이 생성되었습니다: {csv_file}")
    else:
        print("CSV 파일에 저장할 데이터가 없습니다.")

# CSV 파일 생성 호출
create_csv(image_dir, csv_file)
