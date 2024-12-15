import os
import pandas as pd

# 사진 데이터가 저장된 폴더 경로
base_dir = r"C:\Users\USER\jetson_nano_dataset"

# 데이터 저장용 리스트
data = []

# 각 폴더를 순회하며 데이터 수집
for angle_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, angle_folder)
    
    # 폴더인지 확인
    if os.path.isdir(folder_path):
        try:
            # 폴더 이름에서 각도 추출 (폴더 이름이 각도를 나타냄)
            angle = int(angle_folder)  # 폴더 이름이 숫자로 되어 있으므로 이를 각도로 사용
            print(f"각도 폴더: {angle}, 폴더 경로: {folder_path}")  # 디버깅: 각도와 폴더 경로 출력
            
            # 폴더 안에 있는 파일들을 순회
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # 파일이 이미지 파일인지 확인 (확장자가 .jpg 또는 .png인 경우)
                if os.path.isfile(file_path) and file_name.endswith(('.jpg', '.png')):
                    # 각 폴더 안의 모든 이미지는 해당 폴더 이름의 각도를 가지므로 해당 폴더 이름을 각도로 저장
                    data.append({"frame_path": file_path, "angle": angle})
        
        except ValueError:
            print(f"폴더 이름에서 각도를 추출할 수 없습니다: {angle_folder}")

# DataFrame 생성
df = pd.DataFrame(data)

# CSV 파일로 저장
output_csv_path = os.path.join(base_dir, "steering_data.csv")
df.to_csv(output_csv_path, index=False)

print(f"CSV 파일이 생성되었습니다: {output_csv_path}")
