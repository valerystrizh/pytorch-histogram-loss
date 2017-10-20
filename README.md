# Histogram Loss

This is implementation of the paper [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)

After every 10 epochs, model is saved to: `finetuned_histogram_e{}`

## Dataset
[Market-1501 Dataset](http://www.liangzheng.org/Project/project_reid.html)

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

required arguments:
--dataroot DATAROOT   path to dataset
optional arguments:
  -h, --help            show this help message and exit
  --batchSizeTest BATCHSIZETEST     batch size for test and query dataloaders
  --batchSizeTrain BATCHSIZETRAIN     number of data loading workers
  --cuda     number of data loading workers
  --lr LR     number of data loading workers
  --manualSeed MANUALSEED     number of data loading workers
  --nbins NBINS     number of data loading workers
  --nepoch NEPOCH     number of epochs to train
  --nworkers NWORKERS     number of data loading workers
  --out OUT     number of data loading workers
```

    $ # example
    $python main.py --dataroot data/Market-1501-v15.09.15 --cuda --nepoch 150 --out histogram --manualSeed 18
