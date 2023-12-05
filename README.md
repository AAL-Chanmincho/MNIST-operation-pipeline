# MNIST-operation-pipeline
                         

## Environment

1. Python version 3.11.5(pyenv, MNIST-operation-pipeline)
                                

## Quick Start


> For documentation on setting up and running Docker Compose, see [Airflow Docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html).

## Run Docker
```bash
# Set environment variable
cp .env.example .env ## create .env file then fill up variables
```
```bash
# Make sure to initialize Docker settings based on the above airflow documentation
docker compose build
docker compose up
```

## Continuous Training (CT)
> When you predict the number, if the confidence score of predicted value is lower than 50%, 
> input image is uploaded to your s3_bucket under /images directory 

> Then upload the mnlist_label.json to the s3_bucket root directory with format below. 
```json
[
    {"filename": "20231204014102_sample_image.webp", "label": 2},
    {"filename": "20231204014103_sample_image.webp", "label": 7},
    {"filename": "20231204014104_sample_image.webp", "label": 2}
]
```
> Training will be executed based on your labeling and images on s3_bucket. It will be repeated every 30 minutes or you can run on the Airflow UI


### Architecture
![img.png](./docs/images/img.png)
     
#### Links

MLFlow UI
- localhost:5001

Airflow web UI
- localhost:8080

FE
- localhost:4321 

BE
- localhost:8000

Prometheus
- localhost:9090

Grafana
- localhost:5002


<br><br>

## The points you should know

- At first, when you run the project, it retries for 1200 seconds until a model is created in the backend because there is no registered model. so once airflow is running, connect to webserver (localhost:8080) and start the pipeline if it hasn't already been started. 
- Once the ML model is trained by the pipeline automation, it should work fine and run. You can check the automated MNIST application through FE (localhost:4321).


## MNIST datasets
- https://www.kaggle.com/datasets/scolianni/mnistasjpg/

