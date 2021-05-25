## Main code structure

|-- train.py

|-- vis_w.py

|-- utils/

​	|-- utils.py: setup initial hyper-parameters such as full attribute lists

|-- graphs/

​		|-- xray_classifier/

​		|-- stylegen_v2/

​				|-- constants.py: save some constants and  paths for the pertained models

​				|-- stylegan2.py and networks.py: stylegen_v2 model (TODO: update to the latest stylegan2)

​				|-- transform_base.py: (IMPORTANT) contains main code of the paper, including latent vector transformation, loss compilation, model updating, etc.

I'm giving example training and inference commands in the following:

## Train

> python train.py --model stylegan_v2 --transform xray \
>
> --num_samples 20000 --learning_rate 1e-4 --latent w --attrList Cardiomegaly \
>
> --walk_type linear --loss l2 --gpu 2  \
>
> --models_dir ./models_chexperts --overwrite_config 

## Inference

> python vis_w.py models_chexperts/Chexpert_Cardiomegaly_stylegan_v2_xray_linear_lr0.0001_l2_scene_w/opt.yml \
>
> --gpu 2 --noise_seed 12 --num_samples 30 --num_panels 10 \
>
> --save_path_w  models_chexperts/Chexpert_Cardiomegaly_stylegan_v2_xray_linear_lr0.0001_l2_scene_w/model_w_1_walk_module.ckpt 

## ISSUES 

> Make sure the input and output of the pretained classifier is consistant to the one you give in this code, e.g., input: value ranges and normalizations; output: # predicted classes and the attribute index of the desired attribtue should be correctly given.
