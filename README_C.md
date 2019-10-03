＃用于模型压缩的编程系统


Condensa是一种基于Python的可编程模型压缩框架。
它带有一组内置的压缩​​运算符，可针对DNN的特定组合撰写复杂的压缩方案、硬件平台和优化目标。
常见的编程要素例如条件，迭代和递归均能够在本地支持。
为了恢复压缩过程中所丢失的精度，Condensa使用了基于约束模型压缩的优化公式，并使用基于增强拉格朗日的算法作为优化器。


**状态**：Condensa目前正在积极开发中，非常欢迎提供错误报告，pull请求和其他反馈。有关如何贡献的更多详细信息，请参见下面的贡献部分。


##支持的运营商和方案


Condensa提供了以下预构建的压缩方案集：


* [非结构化修剪]（https://nvlabs.github.io/condensa/modules/schemes.html#unstructured-pruning）
* [过滤器和神经元修剪]（https://nvlabs.github.io/condensa/modules/schemes.html#neuron-pruning）
* [块修剪]（https://nvlabs.github.io/condensa/modules/schemes.html#block-pruning）
* [量化]（https://nvlabs.github.io/condensa/modules/schemes.html#quantization）
* [方案组成]（https://nvlabs.github.io/condensa/modules/schemes.html#composition）


上面的方案是使用一个或多个[压缩运算符]（https://nvlabs.github.io/condensa/modules/pi.html）构建的，可以通过多种方式组合以定义自己的自定义方案。


请参阅[文档]（https://nvlabs.github.io/condensa/index.html），以获取有关可用运算符和方案的详细说明。


##初始条件


Condensa要求：


*有效的Linux安装（我们使用Ubuntu 18.04）
* NVIDIA驱动程序和CUDA 10+支持GPU
* Python 3.5或更高版本
* PyTorch 1.0或更高版本


##安装


从Condensa存储库中检索最新的源代码：


```执行代码
git clone https://github.com/NVlabs/condensa.git
```


导航到源代码目录并运行以下命令：


```执行代码
点安装-r requirements.txt
```


要检查安装，请运行单元测试套件：


```执行代码
bash run_all_tests.sh -v
```


＃＃ 入门尝试


[MNIST LeNet5笔记本]（https://github.com/NVlabs/condensa/blob/master/notebooks/LeNet5.ipynb）包含使用Condensa压缩预训练模型的简单分步演练。查看`examples /`文件夹，以获得使用Condensa的其他更复杂的示例。


##文档


文档可在[此处]（https://nvlabs.github.io/condensa/）获得。我们即将发布详细的论文
描述Condensa的开发原因，功能和性能结果。


##贡献


我们感谢所有贡献，包括错误修复，新功能和文档以及其他教程。您可以发起
通过Github pull请求捐款。在进行代码贡献时，请遵循“ PEP 8” Python编码标准并提供
新功能的单元测试。最后，请确保使用-s标志或添加来注销您的提交
提交消息中的“签名后：姓名<电子邮件>”。


##引用Condensa


如果您使用Condensa进行研究，请考虑引用以下论文：


```
@article{condensa2019,
    title = {A Programming System for Model Compression},
    author = {Joseph, Vinu and Muralidharan, Saurav and Garg, Animesh and Garland, Michael},
    journal = {CoRR},
    volume = {}
    year = {2019},
    url = {}
}
```


##免责声明


Condensa是研究原型，不是NVIDIA官方产品。许多功能仍处于试验阶段，尚待继续更新文档。

