import time

from train import Model

if __name__ == "__main__":
    model = Model()

    start_time = time.time()
    model.train()
    end_time = time.time()

    print("Training time: ", end_time - start_time)
