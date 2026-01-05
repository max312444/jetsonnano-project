# 리사이징
resize_images(input_path="C:/jetson_nano_dataset", output_path="C:/change_image")

# 데이터 증강 설정
augmentations = get_augmentations()

# 데이터셋 로딩
dataset = DrivingDataset(
    image_dir="C:/change_image",
    label_file="C:/jetson_csv",
    transform=augmentations
)

# 데이터 분할
train_data, test_data = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_data))
val_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_data))
test_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(test_data))

# 모델 학습
model = NvidiaModel()
train_model(model, train_loader, val_loader, num_epochs=20)
