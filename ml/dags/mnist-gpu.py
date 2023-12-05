from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import redis

# DAG의 기본 인자 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['happyjarban@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date': datetime(2022, 12, 30),
    'retry_delay': timedelta(minutes=5),
}

redis_host = "redis"
redis_port = 6379
redis_password = ""


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.img_labels = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_info = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_info['filename'])
        image = Image.open(img_path).convert('L')
        label = img_info['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def create_model(**kwargs):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    net = Net()
    net.to(device)
    return net


def download_files_from_s3(bucket_name, s3_folder, local_dir, access_key, secret_key):
    s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder).get('Contents', [])
    print(objects)

    if not objects:
        print("No files found in S3 bucket.")
        return False

    for obj in objects:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        path, filename = os.path.split(obj['Key'])
        local_file_path = os.path.join(local_dir, filename)
        s3_client.download_file(bucket_name, obj['Key'], local_file_path)

    return True


def data_preparation(**context):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    bucket_name = os.getenv('S3_BUCKET_NAME')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    image_folder = 'images'
    label_file = 'mnlist_label.json'

    local_image_dir = '/opt/airflow/data/images'
    local_label_file_dir = '/opt/airflow/data'

    r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    run_number = r.get('run_number')
    if run_number is None:
        run_number = 0
    else:
        run_number = int(run_number)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if run_number == 0:
        print("Load default train data")
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        # S3에서 이미지 및 레이블 파일 다운로드
        print("Download Train data from S3")
        images_downloaded = download_files_from_s3(bucket_name, image_folder, local_image_dir, access_key, secret_key)
        labels_downloaded = download_files_from_s3(bucket_name, label_file, local_label_file_dir, access_key,
                                                   secret_key)

        if images_downloaded and labels_downloaded:
            print("Create Custom Data Set")
            trainset = CustomImageDataset(
                annotations_file=f"{local_label_file_dir}/{label_file}",
                img_dir=local_image_dir,
                transform=transform
            )
        else:
            print("Using local MNIST dataset as fallback.")
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    torch.save(trainloader, 'trainloader.pth')
    torch.save(testloader, 'testloader.pth')

    r.set('run_number', run_number+1)

    return {'trainloader_path': 'trainloader.pth', 'testloader_path': 'testloader.pth'}


def train_save_model(**context):
    import torch.optim as optim
    import torch.nn as nn
    import torch
    import mlflow

    data_loaders = context['ti'].xcom_pull(task_ids='data_preparation')
    trainloader_path = data_loaders['trainloader_path']
    trainloader = torch.load(trainloader_path)
    testloader_path = data_loaders['testloader_path']
    testloader = torch.load(testloader_path)

    model = create_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(3):
        with mlflow.start_run(nested=True):  # 중첩 실행 시작
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(trainloader)
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

            # 각 에포크에 대한 파라미터 및 메트릭 로깅
            mlflow.log_param('epoch', epoch)
            mlflow.log_metric('loss', epoch_loss)

    # 모델 평가는 메인 실행에서 수행
    with mlflow.start_run():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct // total
        print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

        # 정확도 로깅
        mlflow.log_metric('accuracy', accuracy)

        # 모델 저장 및 로깅
        mlflow.pytorch.log_model(model, "model", registered_model_name="mnist_pytorch_model")


with DAG(
        'mnist_pytorch_training',
        default_args=default_args,
        description='MNIST training with PyTorch and MLflow',
        schedule_interval=timedelta(minutes=30),
        catchup=False
) as dag:
    import mlflow

    MLFLOW_SERVER_URL = 'http://mlflow-server:5001'
    mlflow.set_tracking_uri(uri=MLFLOW_SERVER_URL)
    mlflow.set_experiment('mnist_pytorch_experiment')

    mlflow.pytorch.autolog(silent=True, log_models=False)

    data_preparation = PythonOperator(
        task_id='data_preparation',
        python_callable=data_preparation,
        dag=dag,
        provide_context=True
    )

    train_and_save_model_job = PythonOperator(
        task_id='train_save_model',
        python_callable=train_save_model,
        dag=dag,
        provide_context=True
    )

    data_preparation >> train_and_save_model_job
