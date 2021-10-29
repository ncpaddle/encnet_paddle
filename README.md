
# EncNet-Paddle : Context Encoding for Semantic Segmentation
 English | [简体中文](README_CN.md)
 
## Introduction 

In this repository, we implement the Contect Encoding Module which can improve semantic segmentation results a lot under the [Paddle](https://www.paddlepaddle.org.cn/) framework. Our model achieves 74.85% mIoU on Cityscapes dataset after 80,000 steps training, which is lower than the results with [mmsegmentation_encnet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet).  We ascribe this substandard precision to the differences between *PaddleSeg* and *mmsegmentation*, which mainly comes from the data augmentation and learning rate schedule method.

- Original Paper : [Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904)  
- Official repo : [Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)  
- Openmmlab : [mmsegmentation_encnet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet)  

## Code Structure
```
encnet_reprod
     ├──  diff # reprod_log的产生的结果
     ├──  encnet_paddle # encnet_Paddle
        ├── config
            ├── encnet
                ├── encnet_cityscapes.yml # 配置文件
        ├── Paddleseg
            ├── models
                ├── backbones
                    ├── resnet_w.py # 复现的主干网络
                ├── losses
                    ├── se_loss.py # 损失函数
                ├── encnet.py # encnet代码
        ├──  train.py  # 训练入口文件
        ├──  val.py # 预测文件
     ├──  encnet_pytorch # encnet_Pytorch

```

## Results

<center>  
     
|Method| Backbone | Crop Size | lr schedule | mIoU | models |    
|----|----|----|----|----|----|       
| Encnet | R-101-D8 | 512x1024 | 80000 | 74.85 | [model link]() |  
     
</center>  

## Reprod log

- [forward_diff.log](./diff/forward_diff.log)  
- [metric_diff.log](./diff/metric_diff.log)  
- [loss_diff.log](./diff/loss_diff.log) 
- [bp_align_diff.log](./diff/bp_align_diff.log)  
- [train_align_diff.log](./diff/train_align_diff.log)  
- [train log](./diff/train.log) 

## Train & Test

To train the model yourself, run :  
```
python -m paddle.distributed.launch train.py \ 
       --config configs/encnet/encnet_cityscapes.yml\ 
       --do_eval \ 
       --use_vdl \ 
       --save_interval 3000 \ 
       --save_dir output/ 
```
To test the results with the model we provided :
```
python val.py \
	--config configs/encnet/encnet_cityscapes.yml \ 
	--model_path output/iter_80000/model.pdparams
```

## AI studio link

* [https://aistudio.baidu.com/aistudio/projectdetail/2535213](https://aistudio.baidu.com/aistudio/projectdetail/2535213)


