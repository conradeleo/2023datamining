# 2023datamining
group7  
廖偉陳：GRU  
朱建誠：LSTM  
流雲得：RNN  
陳重光：CNN + preprocessing  

GRU:
===
```shell
pip install -r requirements
```
```shell
cd src
```

arguments:
----------
`--train_mode`: 決定是要拿`train_1.csv`訓練模型 還是要拿`train_2.csv`評估表現
  不用輸入參數 只要有就是`True` 沒則是`False`, **另一半還沒做好，請全程打開**

**注意：除了`train_1.csv`跟`train_2.csv`我沒讓程式能讀其他檔案，請確保檔案確實在`Dataset`資料夾內**

`--timestep`、`--batch_size`、`--epoch`、`on_gpu` 你知道的

`--n_parallel_process`: 沒用 還沒改掉

ex:
---
```shell
python main.py --train_mode --epoch 10
```
