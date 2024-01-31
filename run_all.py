import time

from matplotlib import pyplot as plt

from train import Model

if __name__ == "__main__":
    bert_resnet_with_concat = Model(model=0)

    start_time = time.time()
    bert_resnet_with_concat.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    bert_resnet_with_attention = Model(model=1)

    start_time = time.time()
    bert_resnet_with_attention.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    bert_densenet_with_concat = Model(model=2)

    start_time = time.time()
    bert_densenet_with_concat.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    bert_densenet_with_attention = Model(model=3)

    start_time = time.time()
    bert_densenet_with_attention.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    resnet = Model(model=0, ablate=1)

    start_time = time.time()
    resnet.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    bert = Model(model=0, ablate=2)

    start_time = time.time()
    bert.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    densenet = Model(model=2, ablate=1)

    start_time = time.time()
    densenet.train()
    end_time = time.time()

    print("Training time:", end_time - start_time)

    plt.plot(bert_resnet_with_concat.train_loss, label="BertResnetWithConcat")
    plt.plot(bert_resnet_with_attention.train_loss, label="BertResnetWithAttention")
    plt.plot(bert_densenet_with_concat.train_loss, label="BertDensenetWithConcat")
    plt.plot(bert_densenet_with_attention.train_loss, label="BertDensenetWithAttention")
    plt.plot(resnet.train_loss, label="Resnet Only")
    plt.plot(bert.train_loss, label="Bert Only")
    plt.plot(densenet.train_loss, label="Densenet Only")
    plt.title("Train Loss")
    plt.legend()
    plt.show()

    plt.plot(bert_resnet_with_concat.val_accuracy, label="BertResnetWithConcat")
    plt.plot(bert_resnet_with_attention.val_accuracy, label="BertResnetWithAttention")
    plt.plot(bert_densenet_with_concat.val_accuracy, label="BertDensenetWithConcat")
    plt.plot(
        bert_densenet_with_attention.val_accuracy, label="BertDensenetWithAttention"
    )
    plt.plot(resnet.val_accuracy, label="Resnet Only")
    plt.plot(bert.val_accuracy, label="Bert Only")
    plt.plot(densenet.val_accuracy, label="Densenet Only")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.show()
