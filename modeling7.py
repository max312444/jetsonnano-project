import os
import cv2

# 원본 이미지 폴더
input_folder = r"C:\Users\USER\jetson_nano_dataset"

# 증강 이미지 저장 폴더
output_folder = r"C:\Users\USER\finish_image"
os.makedirs(output_folder, exist_ok=True)

# 모델에 맞는 크기 (200x66)
target_size = (200, 66)

# 폴더 내 모든 파일 처리
files = os.listdir(input_folder)
print(f"폴더 내 파일들: {files}")  # 폴더 내 파일 목록 출력

for filename in files:
    print(f"\n파일 확인 중: {filename}")  # 파일 확인

    # 확장자 확인 (이미지 파일만 처리)
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"확장자가 올바르지 않거나 이미지 파일이 아닙니다: {filename}")
        continue

    # 파일 경로 생성
    image_path = os.path.join(input_folder, filename)
    print(f"처리 중인 이미지 경로: {image_path}")

    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 이미지 읽기 실패한 경우
    if image is None:
        print(f"이미지를 불러오지 못했습니다: {image_path}")
        # 추가 정보 출력
        if not os.path.exists(image_path):
            print(f"파일이 존재하지 않습니다: {image_path}")
        else:
            print(f"파일 경로 또는 형식에 문제가 있을 수 있습니다: {image_path}")
        continue

    # 이미지 크기 확인
    height, width, _ = image.shape
    print(f"원본 이미지 크기: {width}x{height}")

    # 크기 조정 (모델 입력 크기인 200x66으로 조정)
    resized_image = cv2.resize(image, target_size)
    print(f"리사이즈 후 크기: {resized_image.shape[1]}x{resized_image.shape[0]}")

    # 저장할 파일 경로 생성
    name, ext = os.path.splitext(filename)  # 원본 파일 이름 유지
    output_filename = f"cropped_{name}.jpg"  # 결과 파일은 항상 .jpg 확장자
    output_path = os.path.join(output_folder, output_filename)
    print(f"이미지 저장 경로: {output_path}")

    # 결과 저장
    if cv2.imwrite(output_path, resized_image):
        print(f"전처리된 이미지를 저장했습니다: {output_path}")
    else:
        print(f"이미지 저장 실패: {output_path}")

print("모든 이미지의 전처리가 완료되었습니다.")
