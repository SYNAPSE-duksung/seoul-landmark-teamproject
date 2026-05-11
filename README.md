# seoul-landmark-teamproject: 서울 랜드마크 이미지 분류 및 전이학습 비교 실험

시냅스 2기의 서울 랜드마크 이미지 분류 모델 구현 및 전이학습 비교 실험 팀 과제 레포지토리

서울 랜드마크 이미지 데이터셋을 활용하여 이미지 분류 모델을 구현하고,
기본 CNN 모델과 전이학습(Transfer Learning) 기반 모델의 성능을 비교하는 프로젝트입니다.

PyTorch 기반으로 데이터 전처리, 모델 학습, 검증, 추론 과정을 직접 구현하며
딥러닝 이미지 분류 파이프라인과 전이학습 기법을 학습하는 것을 목표로 합니다.

---

# 과제 목표

* 서울 랜드마크 이미지 분류 모델 구현
* 기본 CNN 모델과 전이학습 모델 성능 비교
* Dataset / DataLoader 구성 및 데이터 전처리 이해
* 모델 구조 및 학습 흐름 이해
* 다양한 성능 개선 실험 수행
* 오분류 분석 및 결과 해석

---

# Branch 구성

본 레포지토리는 팀별 브랜치를 기준으로 관리됩니다.

| Branch   | 설명            |
| -------- | ------------- |
| `main`   | 과제 설명 및 공통 안내 |
| `team-A` | A팀 실험 및 결과    |
| `team-B` | B팀 실험 및 결과    |
| `team-C` | C팀 실험 및 결과    |
| `team-D` | D팀 실험 및 결과    |

각 팀은 별도의 브랜치에서 실험 코드, 보고서, 결과 파일을 관리합니다.

---

# 주요 내용

## 1. 기본 CNN 이미지 분류 모델 구현

* 이미지 데이터 로드
* CustomDataset / DataLoader 구현
* CNN 모델 직접 설계
* 모델 학습 및 검증
* Validation Accuracy 확인
* Test Inference 및 Submission 생성

---

## 2. 전이학습 기반 모델 구현

* torchvision / timm 기반 모델 사용
* pretrained weight 활용
* classifier layer 수정
* freeze / fine-tuning 전략 비교
* CNN baseline 대비 성능 비교


---

## 3. 성능 개선 실험

다양한 실험 중 팀별로 선택을 통해 모델 성능을 비교 및 분석합니다.

### 실험 예시

* 모델 구조 비교
* pretrained 사용 여부 비교
* fine-tuning 전략 비교
* Learning Rate / Batch Size 튜닝
* Optimizer 비교
* 데이터 증강(Augmentation) 비교
* Confusion Matrix 분석
* 오분류 이미지 분석

---

# Dataset 구조

```bash
dataset/
├── train/
├── test/
├── train.csv
├── test.csv
└── sample_submission.csv
```

데이터셋 다운로드:

[Google Drive Dataset](https://drive.google.com/drive/folders/1QZdn2XMspDha_Xd-nmYK--OgMISIrlzV?usp=sharing&utm_source=chatgpt.com)

과제 안내 노션:
[서울 랜드마크 이미지 분류 모델 구현 및 전이학습 비교 실험](https://www.notion.so/35d5595bcb58806faf13c9646fc38a57)

---

# 프로젝트 구조

```bash
project/
├── dataset/  # dataset 폴더 내부의 데이터는 업로드 하지 않아도 좋습니다
├── cnn_baseline.ipynb
├── transfer_learning.ipynb
├── experiments/
├── reports/
├── results/
├── submission/
└── README.md
```

---

# 실행 흐름

```text
데이터 불러오기
→ 이미지 경로 생성
→ Dataset 구성
→ DataLoader 구성
→ 모델 정의
→ 학습
→ 검증
→ 테스트 예측
→ Submission 생성
```

---

# 팀별 보고서

각 팀의 최종 보고서에는 다음 내용이 포함됩니다.

* 데이터 분석
* CNN 모델 구조 설명
* 전이학습 모델 설명
* 실험 결과 비교
* 성능 개선 과정
* 오분류 분석
* 팀원별 역할
* 생성형 AI 사용 기록

---

# 사용 기술

* Python
* PyTorch
* torchvision
* timm
* pandas
* scikit-learn
* matplotlib

---

# 팀 구성

| Team | Members       |
| ---- | ------------- |
| A팀   | 최유경, 박소현, 이지현 |
| B팀   | 신윤서, 이유림, 김예지 |
| C팀   | 박수빈, 김연수, 김소연 |
| D팀   | 신연주, 김민진, 김유진 |

---

# 참고

본 프로젝트는 딥러닝 기반 이미지 분류 및 전이학습 학습을 위한 팀 프로젝트입니다.
