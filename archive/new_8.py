import cv2
import os
import pandas as pd

# 입력 및 출력 폴더 경로
input_folder = r"C:\Users\USER\dataset_0117"  # 원본 이미지 폴더
output_folder = r"C:\Users\USER\augmented_frames"  # 증강 이미지 저장 폴더
os.makedirs(output_folder, exist_ok=True)

# CSV 파일 경로
csv_path = r"C:\Users\USER\data.csv"

# CSV 파일 로드
try:
    df = pd.read_csv(csv_path)
    print("CSV 파일 열 이름:", df.columns.tolist())

    # 열 이름이 정확하지 않을 경우 대체 이름 확인
    if 'steering_angle' in df.columns:
        angle_column = 'steering_angle'
    elif 'angle' in df.columns:
        angle_column = 'angle'
    else:
        raise KeyError("CSV 파일에 'steering_angle' 또는 'angle' 열이 없습니다.")

    # 이미지 전처리 함수
    def preprocess_image(image_path, output_path, angle):
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            return

        # ROI 설정 (상단 30% 제거)
        height = image.shape[0]
        roi = image[int(height * 0.3):, :]

        # 크기 조정 (200x66)
        resized = cv2.resize(roi, (200, 66))

        # 결과 저장 (파일명에 각도 포함)
        output_filename = f"{angle}_cropped_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(output_path, output_filename), resized)

    # 폴더 내 모든 이미지 처리
    for index, row in df.iterrows():
        frame_path = os.path.join(input_folder, row['frame_path'])
        angle = row[angle_column]
        preprocess_image(frame_path, output_folder, angle)

    print("모든 이미지 전처리가 완료되었습니다.")

except FileNotFoundError:
    print(f"CSV 파일이 존재하지 않습니다: {csv_path}")
except KeyError as e:
    print(f"열 이름 오류: {e}")
except Exception as e:
    print(f"예기치 않은 오류가 발생했습니다: {e}")
