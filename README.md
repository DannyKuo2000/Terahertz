# **Introduction:**
This project is a ONN-based autoencoder model implementation. Optical Neural Network (ONN) is a kind of physical neural network computing with physical light. It is composed with multi-layers of diffraction layers, which would create different optical path difference in each physical neuron. And combining with diffraction, ONN　could act as electronic nerual network (ENN) but with less parameters and presentations. 

# Project structure:
```plaintext
Terahertz/
├── model/          # 模型相關的程式碼
│   ├── init.py
│   ├── model.py    # 定義神經網路結構
│   (Not implemented)
│   ├── (loss.py)     # 定義損失函數
│   ├── (utils.py)    # 其他輔助函數
│
├── train.py        # 訓練程式
├── test.py         # 測試/驗證程式
├── data/        # 數據處理相關
│   ├── dataloader.py
│   ├── preprocess.py
├── runs/           # 儲存train.py訓練完的Tensorboard可視化結果
├── checkpoints/    # 儲存train.py訓練完的weights
├── results/        # 儲存test.py測試完的results
(Not implemented)
├── configs/        # 超參數和設定檔
│   ├── config.yaml
├── scripts/        # 可能的執行腳本
│   ├── run_training.sh
│   ├── evaluate.sh
├── logs/           # 訓練時的log文件
```

# **Writing example:**
## Body Text:
```
python main.py --input data.jpg --output result.png
```

**這是加粗文字** __這是加粗文字__
*這是斜體文字* _這是斜體文字_

## Unordered Lists:
- 項目一
- 項目二
    - 子項目一
    - 子項目二

## Ordered Lists:
1. 第一項
2. 第二項
    1. 子項一
    2. 子項二

## Code Blocks
```
def hello_world():
    print("Hello, world!")
```

## Horizontal Lines
可以用三個 - 或 * 來插入水平線來分隔內容。

1234
---
12345