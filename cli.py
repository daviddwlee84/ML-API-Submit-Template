from train import TrainArgs, train_model

if __name__ == "__main__":
    args = TrainArgs().parse_args()
    train_model(args)
