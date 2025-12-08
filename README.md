## 11/27-12/4 ToDo：
1. 各个模型添加输出有害类别序号
2. 将各个模型输入输出封装成接口函数，汇总到统一文件（专家集成模块），根据超参数的权重加权得到最终的prompt score和response score，能跑通整个pipeline
3. 补充统一的环境配置文件requirements.txt


# 环境配置

```bash
conda create -n llmguard python=3.13
conda activate llmguard
pip install -r requirements.txt
```
