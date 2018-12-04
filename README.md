# Histogram Loss

This is implementation of the paper [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf) in PyTorch

## Implementation details

Pretrained resnet 34 is used. [Shared dropout](https://github.com/valerystrizh/pytorch-histogram-loss/blob/master/layers.py#L4) and fully connected layer with 512 neurons are added to the end of the net.

Features should be [l2 normalized](https://github.com/valerystrizh/pytorch-histogram-loss/blob/master/layers.py#L30) before feeding to histogram loss.

[Market-1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) is used for training and testing.

Loss, rank 1 and mAP metrics are visualized using [visdom](https://github.com/facebookresearch/visdom) tools.

## Quality
rank-1: 77.02	

mAP:	54.71

## Usage
Change [config file](https://github.com/valerystrizh/pytorch-histogram-loss/blob/master/config) to set your parameters

```
  --dataroot DATAROOT   path to dataset
  --batch_size BATCH_SIZE
                        batch size for train, default=128
  --batch_size_test BATCH_SIZE_TEST
                        batch size for test and query dataloaders for market
                        dataset, default=64
  --checkpoints_path CHECKPOINTS_PATH
                        folder to output model checkpoints, default="."
  --cuda                enables cuda
  --dropout_prob DROPOUT_PROB
                        probability of dropout, default=0.7
  --lr LR               learning rate, default=1e-4
  --lr_fc LR_FC         learning rate to train fc layer, default=1e-1
  --manual_seed MANUAL_SEED
                        manual seed
  --market              calculate rank1 and mAP on Market dataset; dataroot
                        should contain folders "bounding_box_train",
                        "bounding_box_test", "query"
  --nbins NBINS         number of bins in histograms, default=150
  --nepoch NEPOCH       number of epochs to train, default=150
  --nepoch_fc NEPOCH_FC
                        number of epochs to train fc layer, default=0
  --nworkers NWORKERS   number of data loading workers, default=10
  --visdom_port VISDOM_PORT
                        port for visdom visualization
			

```

    $ #start visdom server
    $ python -m visdom.server -port 8099
    $ python main.py 
