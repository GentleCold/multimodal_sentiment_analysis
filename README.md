## 项目文件

```shell
.
├── dataset
│   ├── data                            // 数据集
│   ├── data_handler.py                 // 数据处理脚本
│   ├── test_without_label.txt          // 测试集
│   └── train.txt                       // 训练集
├── model                               // 定义的各种模型
│   ├── bert_densenet_with_attention.py
│   ├── bert_densenet_with_concat.py
│   ├── bert_resnet_with_attention.py
│   └── bert_resnet_with_concat.py
├── output
│   └── test_with_predict.txt           // 最终的预测输出文件
├── README.md
├── REPORT.md
├── REPORT.pdf                          // 项目报告
├── kaggle.ipynb                        // 在kaggle平台上的运行结果
├── main.py                             // 单个模型训练文件
├── run_all.py                          // 所有模型的训练脚本
└── train.py                            // 训练过程
```

- 实验报告见`REPORT.md/REPORT.pdf`
- 结果文件见`output/test_with_predict.txt`及`kaggle.ipynb`

## 结果复现

通过requirements.txt安装对应包:

```shell
pip install -r requirements.txt
```

由于本地设备原因，报告中的结果展示均在kaggle平台上提供的jupyter环境（使用`GPU P100`）中训练，所有的结果均可在kaggle.ipynb中查看

当然在本地同样也可以复现（需要注意batchsize的大小，默认为16，如果使用GPU训练，需要显存至少为8G，否则需要调小batchsize，这会对结果造成影响）：

1. 一次运行所有模型(默认batch_size为16，learning_rate为1e-5)，复现实验过程：

```shell
python run_all.py
```

2. 也可以运行某一个模型：

```shell
python main.py
```

各参数说明如下

```shell
options:
  -h, --help            show this help message and exit

  --batch_size BATCH_SIZE
                        batch size, defaul=16

  --max_epochs MAX_EPOCHS
                        max number of epochs, default=10

  --model MODEL         0 - Bert Resnet with concat
                        1 - Bert Resnet with attention
                        2 - Bert Densenet with concat
                        3 - Bert Densenet with attention
                        default=0

  --ablate ABLATE       0 - Both txt and img
                        1 - Img only
                        2 - Txt only
                        default=0

  --learning_rate LEARNING_RATE
                        learning rate, default=1e-5
```

示例：

- 运行Bert Resnet with attention 模型：

```shell
python main.py --model 1
```

- 运行Resnet only模型：

```shell
python main.py --ablate 1
```

- 运行Densenet only模型：

```shell
python main.py --model 2 --ablate 1
```

- 运行Bert only模型：

```shell
python main.py --ablate 2
```

## Reference

- https://github.com/guitld/Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis

- https://zhuanlan.zhihu.com/p/381805010

- https://github.com/DA-southampton/NLP_ability
