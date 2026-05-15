# =========================================================
# 1. 라이브러리 불러오기
# =========================================================

import os
import pandas as pd
import numpy as np

from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm


# =========================================================
# 2. CSV 파일 불러오기
# =========================================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

print(train_df.head())
print(test_df.head())


# =========================================================
# 3. 이미지 경로 생성
# ※ 반드시 자신의 폴더 구조에 맞게 수정하기
# =========================================================

# 예시 폴더 구조
# dataset/
# ├── train/
# └── test/

train_df['img_path'] = train_df['image_name'].apply(
    lambda x: f"dataset/train/{x}"
)

test_df['img_path'] = test_df['image_name'].apply(
    lambda x: f"dataset/test/{x}"
)


# =========================================================
# 4. train / validation 데이터 분리
# =========================================================

train_data, val_data = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['label'],
    random_state=42
)

print("Train Data:", len(train_data))
print("Validation Data:", len(val_data))


# =========================================================
# 5. 이미지 전처리 정의
# =========================================================

train_transform = transforms.Compose([

    # 이미지 크기 통일
    transforms.Resize((224, 224)),

    # 데이터 증강
    transforms.RandomHorizontalFlip(),

    # Tensor 형태로 변환
    transforms.ToTensor(),

    # 정규화
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 6. Custom Dataset 정의
# =========================================================

class CustomDataset(Dataset):

    def __init__(self, df, transform=None, is_test=False):

        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        # 이미지 경로 가져오기
        img_path = self.df.iloc[idx]['img_path']

        # 이미지 열기
        image = Image.open(img_path).convert('RGB')

        # transform 적용
        if self.transform:
            image = self.transform(image)

        # test 데이터인 경우 label 없음
        if self.is_test:
            return image

        # train/validation 데이터인 경우 label 반환
        label = self.df.iloc[idx]['label']

        return image, label


# =========================================================
# 7. Dataset 생성
# =========================================================

train_dataset = CustomDataset(
    train_data,
    transform=train_transform
)

val_dataset = CustomDataset(
    val_data,
    transform=val_transform
)

test_dataset = CustomDataset(
    test_df,
    transform=val_transform,
    is_test=True
)


# =========================================================
# 8. DataLoader 생성
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# =========================================================
# 9. CNN 모델 정의
# =========================================================

class CNNModel(nn.Module):

    def __init__(self, num_classes):

        super(CNNModel, self).__init__()

        # 특징 추출 부분
        self.features = nn.Sequential(

            # Conv Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 분류 부분
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 28 * 28, 256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================================================
# 10. Device 설정
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# =========================================================
# 11. 모델 생성
# =========================================================

# 클래스 개수 자동 계산
num_classes = train_df['label'].nunique()

model = CNNModel(num_classes).to(device)

print(model)


# =========================================================
# 12. Loss Function / Optimizer 정의
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================================================
# 13. 모델 학습 및 검증
# =========================================================

epochs = 10

for epoch in range(epochs):

    # -------------------------------
    # train mode
    # -------------------------------

    model.train()

    train_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # gradient 초기화
        optimizer.zero_grad()

        # forward
        outputs = model(images)

        # loss 계산
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # weight 업데이트
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f}")

    # -------------------------------
    # validation mode
    # -------------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # 가장 높은 확률의 클래스 선택
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total

    print(f"Validation Accuracy: {val_accuracy:.4f}")


# =========================================================
# 14. Test 데이터 예측
# =========================================================

model.eval()

predictions = []

with torch.no_grad():

    for images in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())


# =========================================================
# 15. Submission 파일 생성
# =========================================================

sample_submission['label'] = predictions

sample_submission.to_csv(
    "submission_cnn.csv",
    index=False
)

print("\nsubmission_cnn.csv 저장 완료")