from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

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


def data_preparation():
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # 데이터셋에 적용할 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    torch.save(trainloader, 'trainloader.pth')
    torch.save(testloader, 'testloader.pth')

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
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
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
        schedule_interval=timedelta(days=1),
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
    )

    train_and_save_model_job = PythonOperator(
        task_id='train_save_model',
        python_callable=train_save_model,
        dag=dag,
    )

    data_preparation >> train_and_save_model_job
