【状态】：更新中

# 说明    

本项目为ViT的实现,参照了[vit-pytorch](https://github.com/lucidrains/vit-pytorch#)，目的是为了方便中文学习者，如果对你有帮助请给原作者star，谢谢。    


# 目录    
我将按照原作者的实现方式，分别按照simple-vit和vit的实现方式进行构建，然后对每个模块中用到的attention、transformer等也会分别实现，目录如下：

- simple-vit
    - attention.py        构建transformer中的两个核心模块：attention和feedforward
    - transformer.py      构建transformer encoder
    - embedding.py        构建ViT的embedding
    - simple-vit.py       构建ViT
    - test.py             测试ViT

- vit
    - attention.py
    - transformer.py
    - embedding.py
    - vit.py
    - test.py

补充：论文讲解可以看[这里](http://github.com/CliffKai/CS-learning/blob/main/Paper%20Reading/ViT.md)      