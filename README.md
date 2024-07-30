# ML API Submit Template

## Getting Started

```bash
pip install -r requirements.txt
```

## Usage

> Assume each training task only runs on a single GPU

### CLI

> Better used with [`pueue`](https://github.com/Nukesor/pueue)

```bash
# See arguments
python cli.py --help

# Train
python ./cli.py
# Resume
python ./cli.py --resume_run_id 38ef359c0f914a99986a8e6d392e5b13
```

### API

```bash
python api.py
# or
fastapi dev ./api.py
```

http://localhost:8000/docs

> If got error `[WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`, this might mean the port is occupied

### WebUI

```bash
python api.py
# in another terminal
streamlit run ui.py
```

http://localhost:8501/docs

> ### Seem MLFlow result
> 
> Will use `./mlruns`
> 
> ```bash
> mlflow ui --port 8080
> ```

### Pueue

- [Nukesor/pueue: :stars: Manage your shell commands.](https://github.com/Nukesor/pueue)
  1. Download `pueued` and `pueue`
  2. (optional) put them to system path
  3. run `pueued` (optional: make this system service)
  4. check running with `pueue`

```bash
python pueue.py
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
MLFLOW_TRACKING_URI=http://localhost:8080 python api.py
```

```bash
set -a # automatically export all variables
source mlflow_run.env
set +a

# ...
```

- [Artifact Stores — MLflow 2.15.0rc0 documentation](https://mlflow.org/docs/latest/tracking/artifacts-stores.html#amazon-s3-and-s3-compatible-storage)
- [Remote Experiment Tracking with MLflow Tracking Server — MLflow 2.15.0rc0 documentation](https://mlflow.org/docs/latest/tracking/tutorials/remote-server.html#configure-access)

---

## Export MLFlow experiments from one tracking server to another

- [mlflow/mlflow-export-import](https://github.com/mlflow/mlflow-export-import?tab=readme-ov-file)

```bash
# Install
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https:///github.com/mlflow/mlflow-export-import/#egg=mlflow-export-import

# Run local tracking server and export ./mlruns
mlflow server --port 5000
MLFLOW_TRACKING_URI=http://localhost:5000 export-all --output-dir ./export_all
MLFLOW_TRACKING_URI=http://localhost:5000 export-experiment --experiment Default --output-dir ./export_exp
MLFLOW_TRACKING_URI=http://localhost:5000 export-run --run-id 5d9dcca9391643e293e9e3ac6f98b3eb --output-dir ./export_run

# Import experiment to another tracking server
MLFLOW_TRACKING_URI=http://localhost:8080 import-experiment --experiment-name Default --input-dir ./export_exp
MLFLOW_TRACKING_URI=http://localhost:8080 import-all --input-dir ./export_all
MLFLOW_TRACKING_URI=http://localhost:8080 import-run --run-id 5d9dcca9391643e293e9e3ac6f98b3eb --input-dir ./export_run --experiment-name 'Experiment Name [with special char]'
```

> NOTE: currently export-import runs don't keep the `run_id`
>
> - [Is it possible to keep the run id? · Issue #163 · mlflow/mlflow-export-import](https://github.com/mlflow/mlflow-export-import/issues/163)
> - [[FR] Add optional user specified `run_id` in create_run · Issue #12780 · mlflow/mlflow](https://github.com/mlflow/mlflow/issues/12780)
