FROM python:3.11

WORKDIR /mlflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

# MLflow 서버 실행 명령어
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5001"]
