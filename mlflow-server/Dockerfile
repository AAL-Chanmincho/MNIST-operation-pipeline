FROM python:3.11

# mlflow 사용자 생성 및 권한 설정
WORKDIR /home/mlflow/

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5001

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5001
