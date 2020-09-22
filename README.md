# segment-Chinese-and-new-word-discovery-without-dictionary
无字典分词以及新词发现算法

改写自https://github.com/Moonshile/ChineseWordSegmentation

增加功能：

1. 发现新词： 输出不在基础词典里面存在的新词列表

2. 预处理： 按照标点对语料进行分句

非常感谢原算法作者，在每个函数后都贴心的写了详细的注释，在理解程序上对我这种初学者有非常大的帮助！

参数设置：
互信息、频率等参数需要根据语料进行设置，对于新语料，可以先设定为默认值，再观察初步分词的结果以及对应的数据（互信息，频率）调整参数
