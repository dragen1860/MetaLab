#  MetaLab
Universal Meta-Learning Library, with Pytorch GPU Accerleration.

## Algorithms
Currently supported meta-learning algorithms:

- Model-Agnostic Meta-Learning (MAML): https://arxiv.org/abs/1703.03400

- Reptile: https://blog.openai.com/reptile/



# Ominiglot

## Howto
```
python exp/run_omniglot.py
```

## benchmark
| Model                               	| Fine Tune 	| 5-way Acc.    	|               	| 20-way Acc   	|               	|
|-------------------------------------	|-----------	|---------------	|---------------	|--------------	|---------------	|
|                                     	|           	| 1-shot        	| 5-shot        	| 1-shot       	| 5-shot        	|
| MANN                                	| N         	| 82.8%         	| 94.9%         	| -            	| -             	|
| Matching Nets                       	| N         	| 98.1%         	| 98.9%         	| 93.8%        	| 98.5%         	|
| Matching Nets                       	| Y         	| 97.9%         	| 98.7%         	| 93.5%        	| 98.7%         	|
| MAML                                	| Y         	| 98.7+-0.4%    	| 99.9+-0.1%    	| 95.8+-0.3%   	| 98.9+-0.2%    	|
| **Ours**                             	| Y         	| 98.62%     		| 99.52%        	| 96.09%   		| 98.24%    				|



## training curve
![test acc](res/test-acc.png)



# mini-Imagenet


## Howto

download `mini-imagenet` dataset and make it looks like:
```shell
mini-imagenet/
├── images
	├── n0210891500001298.jpg  
	├── n0287152500001298.jpg 
	...
├── test.csv
├── val.csv
└── train.csv

MAML-Pytorch/
├── main.py
├── meta.py
├── Readme.md 
├── naive.md
    ...  
```

and run:
```
python exp/run_omniglot.py
```

## benchmark

| Model                               | Fine Tune | 5-way Acc. |        | 20-way Acc |        |
|-------------------------------------|-----------|------------|--------|------------|--------|
|                                     |           | 1-shot     | 5-shot | 1-shot     | 5-shot |
| Matching Nets                       | N         | 43.56%     | 55.31% | 17.31%     | 22.69% |
| Meta-LSTM                           |           | 43.44%     | 60.60% | 16.70%     | 26.06% |
| MAML                                | Y         | 48.7%      | 63.11% | 16.49%     | 19.29% |
| **Ours**                            | Y         | -      		| - 		| -    		 | - 	|

