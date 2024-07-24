from typing import Optional
from train import TrainArgs, train_model
import mlflow
import mlflow.pytorch
from tap import Tap


class ResumeArgs(Tap):
    resume_run_id: Optional[str] = None
    raise_error_if_checkpoint_not_found: bool = False  # If false, then it will


if __name__ == "__main__":

    print("Tracking URI:", mlflow.get_tracking_uri())
    # The artifact URI is associated with an active run, so you need to start a run first
    # print("Artifact URI:", mlflow.get_artifact_uri())

    resume_args = ResumeArgs().parse_args(known_only=True)

    resume_state_dict = {}
    run_id = None

    if resume_args.resume_run_id:
        client = mlflow.MlflowClient()

        try:
            run = client.get_run(resume_args.resume_run_id)
            arg_dict = mlflow.artifacts.load_dict(
                f"{run.info.artifact_uri}/TrainArgs.json"
            )
            args: TrainArgs = TrainArgs().from_dict(arg_dict)
            if not (
                checkpoints_paths := client.list_artifacts(
                    run.info.run_id, f"checkpoint/latest"
                )
            ):
                if resume_args.raise_error_if_checkpoint_not_found:
                    raise f"Not found checkpoint to resume: {run.info.artifact_uri}/checkpoint/latest"
                print(
                    f"No checkpoint found for run {run.info.run_id}. Will train from scratch."
                )
            else:
                resume_state_dict = mlflow.pytorch.load_state_dict(
                    run.info.artifact_uri + f"/checkpoint/latest"
                )
                run_id = run.info.run_id
        except:
            pass

    if resume_state_dict:
        print(f"Resuming {resume_args.resume_run_id}")
        # NOTE: currently if we resume a run, we don't modify/override the arguments
        args.run_name = None
    else:
        print("Training New Run")
        args = TrainArgs().parse_args(known_only=True)
    train_model(args, run_id=run_id, resume_state_dict=resume_state_dict)
