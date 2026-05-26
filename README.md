# seoul-landmark-teamproject

> 시냅스 2기의 서울 랜드마크 이미지 분류 모델 구현 및 전이학습 비교 실험 팀D 과제 레포지토리
---

## 📌 프로젝트 소개

서울 랜드마크 이미지 데이터셋을 활용하여 이미지 분류 모델을 구현하는 프로젝트입니다.

기본 CNN 모델과 전이학습 기반 모델을 직접 구현하고 성능을 비교합니다.

- 학습 데이터: 723개
- 테스트 데이터: 199개
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

| 실험 | 모델 | LR | Batch Size | Augmentation | Best Val Acc |
|------|------|----|------------|-------------|-------------|
| Exp 1 | 기본 CNN | 0.001 | 32 | 없음 | 93.10% |
| Exp 2 | 데이터 증강 CNN | 0.001 | 32 | Flip, Rotation | 91.03% |
| Exp 3 | 하이퍼파라미터 튜닝 CNN | 0.0001 | 32 | 없음 | 93.10% |
| Exp 4 | ResNet18 전이학습 | 0.001 | 64 | Flip, Rotation | **97.24%** |

### 핵심 결과

- 전이학습(ResNet18)이 **97.24%** 로 가장 높은 성능
- 데이터 증강 적용 시 오히려 성능 하락 (93.10% → 91.03%)
- LR을 낮추니 학습이 더 안정적으로 진행되며 과적합 감소
- 전이학습은 Epoch 2부터 81% 도달로 CNN보다 빠르게 수렴
