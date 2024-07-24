from train import TrainArgs, train_model
import mlflow

if __name__ == "__main__":

    print("Tracking URI:", mlflow.get_tracking_uri())
    # The artifact URI is associated with an active run, so you need to start a run first
    # print("Artifact URI:", mlflow.get_artifact_uri())

    args = TrainArgs().parse_args()
    train_model(args)
