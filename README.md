# ALBERT for Chinese Text Classification


## albert-zh finetune for tensorflow 2.x
基于tf2及albert_zh的中文情感分析任务


## requirements
```
tensorflow 2.3
python 3.8
albert-tiny-zh
```


##label
```
0: negative
1: positive
```

## train
注意调整FLAGS配置
```
python run_classifier.py
```

## infer
```
python predict_from_saved_model.py
```

example
```
"水果新鲜！发货快，服务好，京东物流顶呱呱，快递小哥服务好周到，送货上门，每次都热情满满，辛苦了，必须赞！",
"隔音差，加速有点不给力",
"最满意的是耗电量很低，百公里耗电才11.8度，这样算起来，出行成本比地铁的还低。"
```

output
```
cost time: 0.05300021171569824
['positive', 'negative', 'positive']
```

## reference
> https://github.com/brightmart/albert_zh
> 
> https://github.com/kpe/bert-for-tf2
> 
> https://github.com/kamalkraj/ALBERT-TF2.0


