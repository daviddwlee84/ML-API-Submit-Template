from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Query
import mlflow
from train import TrainTask, train_model, TrainArgs, get_exp_id, get_args_from_model
import config
import utils
from loguru import logger
from pueue import pueue_submit, pueue_logs, pueue_status
from cli import ResumeArgs

PARALLEL_NUM = utils.get_parallel_num()
logger.info(f"Parallel Number: {PARALLEL_NUM}")

app = FastAPI()
# NOTE: somehow start same parameter tasks: using Process will get same result (loss) while using Thread + nested will get different result (loss)
if config.USE_THREAD:
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(
        max_workers=PARALLEL_NUM
    )  # Limit the number of concurrent tasks
else:
    from concurrent.futures import ProcessPoolExecutor

    executor = ProcessPoolExecutor(
        max_workers=PARALLEL_NUM
    )  # Limit the number of concurrent tasks


@app.post("/train")
def submit_training(task: TrainTask, pueue: bool = Query(False)):
    if pueue:
        task_id = pueue_submit(
            get_args_from_model(task), pueue_return_task_id_only=True
        )
        return {
            "message": "Training task has been submitted to pueue",
            "task_id": task_id,
        }

    # https://mlflow.org/docs/latest/tracking/tracking-api.html#launching-multiple-runs
    # https://github.com/mlflow/mlflow/issues/3592
    client = mlflow.MlflowClient()
    # create_run unlike :py:func:`mlflow.start_run`, does not change the "active run" used by :py:func:`mlflow.log_param`.
    run = client.create_run(
        experiment_id=get_exp_id(task.exp_name),
        run_name=task.run_name,
    )
    executor.submit(train_model, task, run.info.run_id)
    return {"message": "Training task has been submitted", "run_id": run.info.run_id}
    # NOTE: start_run cannot handle multiple active runs
    # with mlflow.start_run(run_name=task.run_name) as run:
    #     run_id = run.info.run_id
    #     executor.submit(train_model, task, run_id)
    # return {"message": "Training task has been submitted", "run_id": run_id}


# TODO: resume training
@app.post("/resume")
def resume_training(run_id: str, pueue: bool = Query(False)):
    if pueue:
        task_id = pueue_submit(
            ResumeArgs().parse_args(["--resume_run_id", run_id]),
            pueue_return_task_id_only=True,
        )
        return {
            "message": "Training task has been submitted to pueue",
            "task_id": task_id,
        }

    client = mlflow.MlflowClient()
    try:
        run = client.get_run(run_id)
    except:
        raise HTTPException(
            status_code=404,
            detail=f"Failed to get run. Might be invalid run ID {run_id}",
        )

    if run.info.status == "RUNNING":
        raise HTTPException(
            status_code=400,
            detail=f"Run {run_id} is running.",
        )

    try:
        arg_dict = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/TrainArgs.json")
    except:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load trained argument. Not able to resume.",
        )
    task = TrainArgs().from_dict(arg_dict)
    if not (client.list_artifacts(run.info.run_id, f"checkpoint/latest")):
        raise HTTPException(
            status_code=404,
            detail=f"Not found checkpoint to resume: {run.info.artifact_uri}/checkpoint/latest",
        )
    resume_state_dict = mlflow.pytorch.load_state_dict(
        run.info.artifact_uri + f"/checkpoint/latest"
    )
    executor.submit(train_model, task, run.info.run_id, resume_state_dict)
    return {"message": "Training task has been resumed", "run_id": run.info.run_id}


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


@app.get("/pueue/{mode}/{task_id}")
def get_pueue_task_status(
    mode: Literal["status", "logs", "running_status", "output"],
    task_id: Optional[str] = None,
):
    try:
        if mode == "status":
            return pueue_status(task_id=task_id)
        elif mode == "logs":
            return pueue_logs(task_id=task_id)
        elif mode == "running_status":
            try:
                assert task_id
                status = pueue_logs(task_id=task_id)["task"]["status"]
                if isinstance(status, dict):
                    # {'detail': 'Not Found'}
                    return (
                        "Success"
                        if "Done" in status
                        else status
                    )
                elif isinstance(status, str):
                    return status
                else:
                    raise ValueError(f"Unknown status {status}")
            except:
                raise HTTPException(
                    status_code=400,
                    detail="In running_status mode you should query for an existing task_id",
                )
        elif mode == "output":
            try:
                assert task_id
                log = pueue_logs(task_id=task_id)
                return dict(
                    output=log["output"],
                    is_finished="Done" in log["task"]["status"],
                )
            except:
                raise HTTPException(
                    status_code=400,
                    detail="In output mode you should query for an existing task_id",
                )
        else:
            raise NotImplementedError(f"Unknown mode {mode}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Could not retrieve run status: {e}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
