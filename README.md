# nn

一个手写的简单神经网络，包括
- 层、网络以及组合
- 前向传播、反向传播
- 自定义激活函数
- 自定义损失函数
- 多重梯度下降策略
- 清晰的训练过程信息输出
- 简单的配置以及训练

> 目前现成的是一个用`Minst`数据集训练的网络，使用`relu`激活函数，`softmax`损失函数，`adam`梯度下降策略，`[784, 256, 128, 10]`的网络结构

## 运行要求
从[这里](https://gitcode.com/open-source-toolkit/2be96)下载`Minst`数据集放在项目目录下。

安装[`uv`](https://hellowac.github.io/uv-zh-cn/getting-started/installation/)python包管理工具，运行`uv sync`。

修改`main.py`中的数据集文件路径。
```python
    training_data, training_labels = load_mnist_data('./dataset/minst/train', './dataset/minst/train_labs.txt')
    validation_data, validation_labels = load_mnist_data('./dataset/minst/test', './dataset/minst/test_labs.txt')
```

最后运行`uv run main.py`

## 示例
`nn/main.py`
```python
def main():
    training_data, training_labels = load_mnist_data('./dataset/minst/train', './dataset/minst/train_labs.txt')
    validation_data, validation_labels = load_mnist_data('./dataset/minst/test', './dataset/minst/test_labs.txt')

    network = NetWork(
        [784, 256, 128, 10],
        optimizer_type='Adam',
        optimizer_config={
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
        }
    )

    network.train(training_data, training_labels, validation_data, validation_labels, epochs=25)

if __name__ == '__main__':
    main()
```
输出：
```commandline
Training starts
Epoch 1/25
Train loss: 0.3765
Train accuracy: 0.8894
Validation loss: 0.2500
Validation accuracy: 0.9311
Epoch time: 15.58 seconds
-------------------------
Epoch 2/25
Train loss: 0.2325
Train accuracy: 0.9336
Validation loss: 0.2143
Validation accuracy: 0.9377
Epoch time: 15.40 seconds
-------------------------

...

-------------------------
Epoch 25/25
Train loss: 0.0319
Train accuracy: 0.9916
Validation loss: 0.0763
Validation accuracy: 0.9760
Epoch time: 16.66 seconds
-------------------------
Training complete
```