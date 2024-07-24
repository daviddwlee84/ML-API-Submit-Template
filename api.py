from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import mlflow
from train import TrainTask, train_model


app = FastAPI()
# TODO: set max_workers to be number of GPU / if CPU then no limit..?
executor = ThreadPoolExecutor(max_workers=4)  # Limit the number of concurrent tasks


@app.post("/train")
def submit_training(task: TrainTask):
    # BUG: currently if there is any active run then it will conflict
    # Alternatives
    client = mlflow.MlflowClient()
    run = client.create_run(
        experiment_id=mlflow.tracking.fluent._get_experiment_id(),
        run_name=task.run_name,
    )
    executor.submit(train_model, task, run.info.run_id)
    return {"message": "Training task has been submitted", "run_id": run.info.run_id}
    # with mlflow.start_run(run_name=task.run_name) as run:
    #     run_id = run.info.run_id
    #     executor.submit(train_model, task, run_id)
    # return {"message": "Training task has been submitted", "run_id": run_id}


@app.get("/status/{run_id}")
def get_task_status(run_id: str):
    try:
        run = mlflow.get_run(run_id)
        return {
            "run_id": run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not retrieve run status: {e}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
