# Jetson Nano 기반 자율주행 RC 자동차 프로젝트

본 저장소는 NVIDIA Jetson Nano를 활용하여 자율주행 RC 자동차를 구현한 프로젝트입니다.  
카메라로 인식한 주행 라인을 기반으로 CNN 모델을 학습하고, 이를 통해 조향각과 속도를 자동으로 제어하는 것을 목표로 합니다.

임베디드 환경에서의 컴퓨터 비전 처리, 딥러닝 모델 학습, 실시간 추론 및 하드웨어 제어를 통합적으로 다루는 실습 중심 프로젝트입니다.

---

## GitHub 활동

### GitHub Trophy
<p align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=max312444&theme=gruvbox_purple&no-frame=true&no-bg=true"/>
</p>

### Contribution Graph
<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=max312444&theme=high-contrast"/>
</p>

---

## 🎥 주행 시연 영상 (Driving Demo)

본 영상은 NVIDIA Jetson Nano 환경에서  
CNN 기반 End-to-End 모델을 이용해 카메라 입력만으로  
조향각과 속도를 제어하는 실제 주행 결과를 보여줍니다.

---

## 프로젝트 개요

- 플랫폼: NVIDIA Jetson Nano  
- 주행 방식: 카메라 기반 라인 트래킹  
- 제어 방식: CNN 기반 End-to-End 학습  
- 개발 목적: 임베디드 AI 자율주행 시스템의 전체 파이프라인 구현

- [주행영상1](videos/주행영상1.mp4)
- [주행영상2](videos/주행영상2.mp4)

---

## 시스템 구성

### 하드웨어

- NVIDIA Jetson Nano  
- 카메라 모듈  
- DC 모터  
- 서보 모터  
- DC-DC 컨버터  
- RC 자동차 섀시

### 소프트웨어

- Python  
- PyTorch  
- OpenCV  
- CNN 기반 딥러닝 모델  
- Jetson Nano GPIO 제어

---

## 주요 기능

- 카메라 영상 기반 라인 인식
- 주행 이미지 데이터 수집
- 이미지 전처리 및 데이터셋 구성
- CNN 모델 학습을 통한 조향각 예측
- 예측 결과를 이용한 실시간 조향 및 속도 제어
- Jetson Nano 환경에서의 실시간 추론

---

## 개발 과정에서의 핵심 고려사항

- 조명 변화에 강한 주행을 위해 이미지 전처리 파이프라인 개선
- 규칙 기반 제어 대신 CNN 기반 End-to-End 학습 방식 적용
- Jetson Nano의 제한된 연산 자원을 고려한 모델 구조 및 입력 해상도 조정
- 실시간 추론 성능과 주행 안정성의 균형을 고려한 설계

---

## 알려진 문제 (Known Issues)

- Jetson Nano 환경에서 GPIO 핀 인식이 불안정한 문제
- 특정 배터리 상태에서 DC 모터 출력이 일정하지 않은 현상
- 모델 학습 및 추론 속도가 느린 문제  
  - 향후 CUDA 및 TensorRT 기반 최적화 예정

---

## 향후 개선 사항

- TensorRT를 이용한 추론 속도 최적화
- 다양한 주행 환경 데이터 추가 수집
- 장애물 인식 기능 추가
- 주행 결과 로그 및 시각화 기능 구현

---

## 참고

본 프로젝트는 자율주행 시스템의 전체 흐름  
(센싱, 인식, 판단, 제어)을 임베디드 환경에서 직접 구현하는 것을 목표로 합니다.
