# Histogram Loss

This is implementation of the paper [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf) in PyTorch

## Implementation details

Pretrained resnet 34 is used. Shared dropout with 0.5 probability and one fully connected layer with 512 neurons are added to the end of the net.

Features should be [l2 normalized](https://github.com/valerystrizh/pytorch-histogram-loss/blob/master/layers.py#L30) before feeding to histogram loss.

[Market-1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) is used for training and testing.

Loss, rank 1 and mAP metrics are visualized using [visdom](https://github.com/facebookresearch/visdom) tools.

## Quality
rank-1: 77.02	

mAP:	54.71

## Usage
```
main.py [-h] 
	--dataroot DATAROOT
	[--batch_size BATCH_SIZE]
	[--batch_size_test BATCH_SIZE_TEST]
	[--checkpoints_path CHECKPOINTS_PATH] 
	[--cuda] 
	[--dropout_prob DROPOUT_PROB] 
	[--lr LR]
	[--lr_init LR_INIT] 
	[--manual_seed MANUAL_SEED] 
	[--market]
	[--nbins NBINS] 
	[--nepoch NEPOCH] 
	[--nworkers NWORKERS]
	[--visdom_port VISDOM_PORT]

required argument:
  --dataroot 		DATAROOT   		path to dataset
  
optional arguments:
  -h, --help					show this help message and exit
  --batch_size  	BATCH_SIZE 		batch size for train, default=128
  --batch_size_test 	BATCH_SIZE_TEST 	batch size for test and query dataloaders for market dataset, default=64
  --checkpoints_path 	CHECKPOINTS_PATH	folder to output model checkpoints, default="."
  --cuda                			enables cuda
  --dropout_prob 	DROPOUT_PROB		probability of dropout, default=0.5
  --lr 			LR               	learning rate, default=1e-4
  --lr_init 		LR_INIT     		learning rate for first stage of training only fc layer, default=1e-2
  --manual_seed 	MANUAL_SEED		manual seed
  --market              			calculate rank1 and mAP on Market dataset; dataroot should contain folders "bounding_box_train", "bounding_box_test", "query"
  --nbins 		NBINS         		number of bins in histograms, default=150
  --nepoch 		NEPOCH       		number of epochs to train, default=150
  --nworkers 		NWORKERS   		number of data loading workers, default=10
  --visdom_port 	VISDOM_PORT		port for visdom visualization
```

    $ #start visdom server if use visdom_port argument
    $ python -m visdom.server -port 8099
    $ python main.py --dataroot data/Market-1501-v15.09.15 --cuda --market --checkpoints_path histogram --manual_seed 18 --visdom_port 8099
