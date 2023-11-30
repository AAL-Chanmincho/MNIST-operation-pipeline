FROM apache/airflow:2.7.3-python3.11

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080 8080
