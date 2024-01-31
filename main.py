import argparse
import time

from train import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size, defaul=16"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="max number of epochs, default=10"
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="0 - Bert Resnet with concat\n1 - Bert Resnet with attention\n2 - Bert Densenet with concat\n3 - Bert Densenet with attention\ndefault=0",
    )
    parser.add_argument(
        "--ablate",
        type=int,
        default=0,
        help="0 - Both txt and img\n1 - Img only\n2 - Txt only\ndefault=0",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate, default=1e-5",
    )

    args = parser.parse_args()

    model = Model(
        args.max_epochs, args.learning_rate, args.batch_size, args.model, args.ablate
    )

    start_time = time.time()
    model.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)
    # model.save_test_result()
