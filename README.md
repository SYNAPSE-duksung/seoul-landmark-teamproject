# seoul-landmark-teamproject

> 시냅스 2기의 서울 랜드마크 이미지 분류 모델 구현 및 전이학습 비교 실험 팀D 과제 레포지토리
---

## 📌 프로젝트 소개

서울 랜드마크 이미지 데이터셋을 활용하여 이미지 분류 모델을 구현하는 프로젝트입니다.

기본 CNN 모델과 전이학습 기반 모델을 직접 구현하고 성능을 비교합니다.

- 학습 데이터: 724개
- 테스트 데이터: 200개
- 클래스 수: 10개

---

## 👥 팀원 및 역할

| 이름 | 역할 | 담당 내용 |
|------|------|------|
| 신연주 | 팀장 / 팀원 A | 데이터 로드, 경로 생성, CustomDataset, DataLoader, 전처리, 데이터 증강 비교 실험 |
| 김민진 | 팀원 B | CNN 모델 정의, 학습/검증 루프, test 예측, 하이퍼파라미터 튜닝 실험 |
| 김유진 | 팀원 C | 전이학습 모델 구현, test 예측, 전이학습 방식 비교 실험 |

---

## 🛠️ 기술 스택

| 분류 | 사용 기술 |
|------|------|
| 언어 | Python 3.10 |
| 딥러닝 | PyTorch, torchvision |
| 데이터 처리 | pandas, numpy, scikit-learn |
| 이미지 처리 | PIL (Pillow) |
| 개발 환경 | Google Colab |
| 버전 관리 | Git, GitHub |

---

## 📁 파일 구조

```
seoul-landmark-teamproject/
├── cnn_baseline.ipynb        # 기본 CNN 모델 (A파트 + B파트)
├── transfer_learning.ipynb   # 전이학습 모델 (C파트)
├── submission_cnn.csv        # CNN 모델 예측 결과
├── submission_transfer.csv   # 전이학습 모델 예측 결과
├── D팀_보고서.pdf            # 최종 보고서
├── 발표자료.pdf              # 발표 슬라이드
└── README.md
```

---

## ▶️ 실행 방법

1. 구글 드라이브에 데이터셋 업로드
```
내 드라이브/synaps_team_project
├── train/
├── test/
├── train.csv
├── test.csv
└── sample_submission.csv
```

2. Google Colab에서 노트북 열기

3. 구글 드라이브 마운트
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. 순서대로 셀 실행

---

## 📊 실험 결과

추후 업데이트 예정

| 실험 | 모델 | Val Accuracy |
|------|------|------|
| Exp 1 | 기본 CNN | - |
| Exp 2 | 전이학습 | - |
| Exp 3 | 하이퍼파라미터 튜닝 | - |
| Exp 4 | 데이터 증강 | - |
