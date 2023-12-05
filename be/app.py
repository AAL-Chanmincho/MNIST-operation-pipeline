from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from PIL import Image
import time
import torch
from torchvision import transforms
import mlflow.pytorch
import boto3

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

MLFLOW_SERVER_URL = os.environ.get('MLFLOW_SERVER_URL', 'http://0.0.0.0:5001')
model_uri = "models:/mnist_pytorch_model/latest"


# 모델 로딩을 위한 재시도 함수
def load_model_with_retry(retry_count=60, wait_seconds=20):
    for _ in range(retry_count):
        try:
            mlflow.set_tracking_uri(uri=MLFLOW_SERVER_URL)
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()
            return model
        except Exception as e:
            print(f"모델 로딩 실패: {e}, 재시도 중...")
            time.sleep(wait_seconds)
    raise Exception("모델 로딩에 실패했습니다.")


# FastAPI 애플리케이션 시작 시 모델 로드
@app.on_event("startup")
def startup_event():
    global model
    model = load_model_with_retry()


# 이미지 전처리 함수
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def save_image_to_s3(image_bytes, filename, bucket_name, access_key, secret_key):
    # 클라이언트 생성
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    # 파일 이름 생성 (현재 시간을 포함하여 고유하게 만듬)
    image_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"

    # S3 버킷에 이미지 저장
    try:
        s3_client.put_object(Body=image_bytes, Bucket=bucket_name, Key=f"images/{image_name}")
        print(f"Image {image_name} saved to S3 bucket {bucket_name}")
    except Exception as e:
        print(f"Error saving image to S3: {e}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 이미지 받아오기
    image_bytes = await file.read()

    # 이미지 전처리 및 예측
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    max_values, predicted = torch.max(outputs.data, 1)
    print(outputs.data)
    bucket_name = os.getenv('S3_BUCKET_NAME')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    if max_values.item() < 0:
        save_image_to_s3(image_bytes, file.filename, bucket_name, access_key, secret_key)
    prediction = predicted[0].item()

    # 결과 반환
    return JSONResponse(content={"prediction": prediction})
