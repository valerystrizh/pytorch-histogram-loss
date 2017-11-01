# Histogram Loss

This is implementation of the paper [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf) in PyTorch

Features should be l2 normalized before feeding to histogram loss. Use module L2Normalization from layers.py.

## Implementation details

Pretrained resnet 34 was used. Shared dropout with 0.2 probability and one fully connected layer with 512 neurons were added to the end of the net.

After every 10 epochs, model is saved to `finetuned_histogram_e{}`. Statistics on loss, rank1 and mAP is saved in `loss_statistics.csv` and `rank1map_statistics.csv` files respectively.

[Market-1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) is used.

Method ranks_map from class Evaluation counts statistics on rank1 and mAP. 

## Quality
rank-1: 77.02	

mAP:	54.71

## Usage
```
main.py --dataroot DATAROOT 
	[--batchSizeTest BATCHSIZETEST]
	[--batchSizeTrain BATCHSIZETRAIN] 
	[--cuda] 
	[--lr LR]
	[--manualSeed MANUALSEED] 
	[--nbins NBINS] 
	[--nepoch NEPOCH] 
	[--nworkers NWORKERS]
	[--out OUT] 

required argument:
  --dataroot DATAROOT   		path to dataset
  
optional arguments:
  -h, --help				show this help message and exit
  --batchSizeTest BATCHSIZETEST		batch size for test and query dataloaders
  --batchSizeTrain BATCHSIZETRAIN	batch size for train
  --cuda				enables cuda
  --lr LR				learning rate
  --manualSeed MANUALSEED     		manual seed
  --nbins NBINS     			number of bins in histograms
  --nepoch NEPOCH     			number of epochs to train
  --nworkers NWORKERS     		number of data loading workers
  --out OUT     			folder to output model checkpoints
```

    $ python main.py --dataroot data/Market-1501-v15.09.15 --cuda --nepoch 150 --out histogram --manualSeed 18
