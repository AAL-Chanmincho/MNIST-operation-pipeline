from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import torch
from torchvision import transforms
import mlflow.pytorch

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# MLflow를 사용하여 모델 로드
model_uri = "models:/mnist_pytorch_model/latest"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 이미지 받아오기
    image_bytes = await file.read()

    # 이미지 전처리 및 예측
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    prediction = predicted[0].item()

    # 결과 반환
    return JSONResponse(content={"prediction": prediction})
