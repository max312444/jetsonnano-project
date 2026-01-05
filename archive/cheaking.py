import cv2
import os

def load_image(image_path):
    try:
        # 경로 구분자 정리
        image_path = os.path.normpath(image_path)  # 경로 구분자 통일

        # 이미지 로드
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        # 이미지 크기 확인 (옵션)
        print(f"이미지 크기: {image.shape}")

        return image

    except Exception as e:
        print(f"이미지 로딩 실패: {e}")
        return None

# 예시 경로
image_path = r"C:/Users/USER/Desktop/lastdance/augmented_frames/150_image_20250117_010729_229.jpg"
image = load_image(image_path)

if image is not None:
    print("이미지가 정상적으로 로드되었습니다.")
else:
    print("이미지 로딩에 실패했습니다.")
