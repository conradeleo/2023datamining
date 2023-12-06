# 2023datamining
group7  
廖偉陳：GRU  
朱建誠：LSTM  
流雲得：RNN  
陳重光：CNN + preprocessing  

GRU:
pip install -r requirements
cd src

arguments:
--train_mode: 決定是要拿train_1.csv訓練模型 還是要拿train_2.csv評估表現
  不用輸入參數 只要有就是True 沒則是False, 另一半還沒做好，請全程打開
--timestep
--batch_size
--epoch
--n_parallel_process: 沒用

ex:
python main.py --train_mode --epoch 10
