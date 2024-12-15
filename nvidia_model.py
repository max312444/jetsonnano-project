import torch
import torch.nn as nn

class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # 첫 번째 합성곱 레이어
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),  # 두 번째 합성곱 레이어
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),  # 세 번째 합성곱 레이어
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # 네 번째 합성곱 레이어
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 다섯 번째 합성곱 레이어
            nn.ReLU()
        )
        
        # 입력 이미지 크기 (예: 160x320)
        # Conv2d 레이어를 지나면서 크기가 줄어들고, 마지막에 Flatten을 적용할 때 크기를 계산해야 함
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 15, 100),  # 여기서 크기는 이미지 크기와 Conv2d 파라미터에 맞게 조정
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 최종적으로 조향 각도 출력
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten: 배치 크기 제외한 나머지 차원을 일렬로 펴기
        x = self.fc_layers(x)
        return x
