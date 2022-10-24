# Event Clue Prediction

|事件重要性|事件规模|事件影响|事件强度|事件程度|
|:-----:|:-----:|:-----:|:-----:|:-----:|
| -5 | 小规模 | 正向 | 低强度 | 低 |
| -3 | 中规模 | 负向 | 中强度 | 中 |
| -1 | 大规模 |  | 高强度 | 高 |
| 1 |  |  |  | 最高 |
| 3 |  |  |  |  |
| 5 |  |  |  |  |

## 事件重要性打分(1-3-5打分方式，6分类)

## 事件规模(3分类)

## 事件影响(2分类)

## 事件强度(3分类)

## 事件程度(4分类)


时间线索训练：(train/trainer.py)
```
python trainer.py --flag_id [data_flag] --metrics [train_metrics]
```
where the range of ``data_flag`` is ``[0,1,2,3,4]`` ，the range of ``train_metrics`` is ``['macro', 'micro']``


时间线索预测：(predict/predicter.py)
```
flags = ['score', 'scale', 'influence', 'strength', 'degree']
res_id = event_clue_predict(text, flag_id)
```
where ``text`` is ``输入的事件文本``，``flag_id`` is ``数据的类型``