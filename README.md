# Introduction:
This project is a ONN-based autoencoder model implementation. Optical Neural Network (ONN) is a kind of physical neural network computing with physical light. It is composed with multi-layers of diffraction layers, which would create different optical path difference in each physical neuron. And combining with diffraction, ONN could act as electronic nerual network (ENN) but with less parameters and presentations. 



# Optical Equipment Spec:
- Camera:  https://www.ino.ca/en/solutions/thz/microxcam-384i-thz/
    - Name: MICROXCAM-384i-THz Terahertz Camera (INO)
    - Wavelength range: 70–3189 µm/4,25–0,094 THz
    - Resolution: 384 x 288 pixels
    - Pitch: 35 µm
    - Output: 16-bit raw data, Gigabit Ethernet
    - Weight: 360 g / 0.8 lb  
    - Size: 61 x 61 x 65 mm/2.4 x 2.4 x 2.6 in
    - Frequency: 50 Hz



# Project structure: (maybe delete later)
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