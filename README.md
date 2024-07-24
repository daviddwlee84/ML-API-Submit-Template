# ML API Submit Template

## Getting Started

```bash
pip install -r requirements.txt
```

## Usage

### CLI

> Better used with [`pueue`](https://github.com/Nukesor/pueue)

```bash
python cli.py --help
```

### API

```bash
python api.py
```

http://localhost:8000/docs

### WebUI

```bash
python api.py
# in another terminal
streamlit run ui.py
```

http://localhost:8501/docs

### Seem MLFlow result

> Will use `./mlruns`

```bash
mlflow ui
```

---

Use SQLite

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# ... do some training
MLFLOW_TRACKING_URI=sqlite:///mlruns.db python cli.py

mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```


## Additional Setup

By default MLFlow can be run without [Tracking Server](https://mlflow.org/docs/latest/tracking.html#tracking-server)

![MLFlow tracking](https://mlflow.org/docs/latest/_images/tracking-setup-overview.png)

But according to experience, if MLFlow runs without database and if we have tons of runs, it will cause MLFlow core crash while it will always go through files in `./mlruns` which is heavy.

### MLFlow Tracking + Artifact Server

> - [Official MLflow Docker Image — MLflow 2.15.0rc0 documentation](https://mlflow.org/docs/latest/docker.html)
> - [ubuntu/mlflow - Docker Image | Docker Hub](https://hub.docker.com/r/ubuntu/mlflow)

- [Setting up a Development Machine with MLFlow and MinIO](https://blog.min.io/setting-up-a-development-machine-with-mlflow-and-minio/)
  - [blog-assets/mlflow-minio-setup at main · minio/blog-assets](https://github.com/minio/blog-assets/tree/main/mlflow-minio-setup?ref=blog.min.io)
- [Remote Experiment Tracking with MLflow Tracking Server — MLflow 2.12.1 documentation](https://mlflow.org/docs/latest/tracking/tutorials/remote-server.html#create-compose-yaml)
- [sachua/mlflow-docker-compose: MLflow deployment with 1 command](https://github.com/sachua/mlflow-docker-compose)
- [Toumash/mlflow-docker: Ready to run docker-compose configuration for ML Flow with Mysql and Minio S3](https://github.com/Toumash/mlflow-docker)

```bash
docker compose --env-file mlflow_config.env down
docker compose --env-file mlflow_config.env up -d --build
```

```bash
# Will use local artifact
MLFLOW_TRACKING_URI=http://localhost:8080 python cli.py
```

```bash
set -a # automatically export all variables
source mlflow_run.env
set +a

# ...
```

- [Artifact Stores — MLflow 2.15.0rc0 documentation](https://mlflow.org/docs/latest/tracking/artifacts-stores.html#amazon-s3-and-s3-compatible-storage)
- [Remote Experiment Tracking with MLflow Tracking Server — MLflow 2.15.0rc0 documentation](https://mlflow.org/docs/latest/tracking/tutorials/remote-server.html#configure-access)
