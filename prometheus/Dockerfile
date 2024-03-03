# 사용할 Prometheus의 베이스 이미지
FROM prom/prometheus:v2.30.0

# prometheus.yml 구성 파일을 이미지에 복사
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Prometheus 서버가 사용할 포트 (기본값: 9090)
EXPOSE 9090

# Prometheus 실행 명령어 (이미지에 기본적으로 포함된 명령어를 사용할 수도 있습니다)
CMD ["--config.file=/etc/prometheus/prometheus.yml", \
     "--storage.tsdb.path=/prometheus", \
     "--web.console.libraries=/usr/share/prometheus/console_libraries", \
     "--web.console.templates=/usr/share/prometheus/consoles"]
