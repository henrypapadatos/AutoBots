# Multimodal joint trajectory prediction - Deep learning for Autonomous Vehicles EPFL
This repository is a fork of the [official implementation](https://github.com/roggirg/AutoBots) of the [AutoBots](https://arxiv.org/abs/2104.00563) architectures.

## Contribution Overview

### Optimizer and activation function 
A first potential amelioration is to use the AdamW optimizer instead of the classic Adam one. Opposed to Adam, AdamW does not put the weight decay term in the moving average. This change [often results](https://towardsdatascience.com/why-adamw-matters-736223f31b5d#:~:text=The%20authors%20show%20experimentally%20that,stochastic%20gradient%20descent%20with%20momentum.) in a better training loss and in less overfitting. 

Another improvement that can be tested is to replace the ReLU activation function with GELU, as GELU's smoother nature allows the model to [better capture](https://www.saltdatalabs.com/blog/deep-learning-101-transformer-activation-functions-explainer-relu-leaky-relu-gelu-elu-selu-softmax-and-more) complex patterns within the data.

### Learned positional encoding
The positional encoding implemented in the AutoBot arcitecture is the classic sinusoidal positional encoding used by vaswani et al. in the first transformer architecture. With _t_ being the position of the input token in the sequence and _d_ being the encoding dimension. The sinusoidal positional encoding $\overrightarrow{p_t}$ can be written as follow: 

$$
{\overrightarrow{p_t}}^{(i)}=f(t)^{(i)}:= \begin{cases}\sin \left(\omega_k \cdot t\right), & \text { if } i=2 k \\\ \cos \left(\omega_k \cdot t\right), & \text { if } i=2 k+1\end{cases}
$$

where
$$\omega_k=\frac{1}{10000^{2 k / d}}$$

Here is a vizualisation if the sinusoidal positional encoding as presented in [this](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) blogpost:
![Screenshot 2023-05-28 124239](https://github.com/henrypapadatos/AutoBots/assets/63106608/6afe4746-de46-4541-af73-45025775e50a)

Alternatively, the positional encoding can be learned as a model paramater. This approach is becoming more and more popular and we have therefore deciced to test it in the Autobot architecture. 

### Multi-layer loss

## Description of the dataset 

## Results

## Conclusion

## Difficulties encountered



## Here is a copy of the official AutoBot REAMDE file for installation, setup of datasets, training and evaluation: 

We include support for the following datasets:
- [nuScenes](https://www.nuscenes.org/nuscenes) (ego-agent, multi-agent)
- [Argoverse](https://www.argoverse.org/av1.html) (ego-agent)
- [TrajNet++](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge) Synthetic dataset (multi-agent)
- [Interaction-Dataset](https://interaction-dataset.com/) (multi-agent)

Visit our [webpage](https://fgolemo.github.io/autobots/) for more information.

### Getting Started

1. Create a python 3.7 environment. I use Miniconda3 and create with 
`conda create --name AutoBots python=3.7`
2. Run `pip install -r requirements.txt`

That should be it!

### Setup the datasets

Follow the instructions in the READMEs of each dataset (found in `datasets`).

### Training an AutoBot model

All experiments were performed locally on a single GTX 1080Ti.
The trained models will be saved in `results/{Dataset}/{exp_name}`. 

#### nuScenes
Training AutoBot-Ego on nuScenes while using the raw road segments in the map:
```
python train.py --exp-id test --seed 1 --dataset Nuscenes --model-type Autobot-Ego --num-modes 10 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-lanes True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/nuscenes_h5_files
```

Training AutoBot-Ego on nuScenes while using the Birds-eye-view image of the road network:
```
python train.py --exp-id test --seed 1 --dataset Nuscenes --model-type Autobot-Ego --num-modes 10 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-image True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/nuscenes_h5_files
```

Training AutoBot-Joint on nuScenes while using the raw road segments in the map:
```
python train.py --exp-id test --seed 1 --dataset Nuscenes --model-type Autobot-Joint --num-modes 10 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-lanes True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/nuscenes_h5_files
```

#### Argoverse
Training AutoBot-Ego on Argoverse while using the raw road segments in the map:
```
python train.py --exp-id test --seed 1 --dataset Argoverse --model-type Autobot-Ego --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-lanes True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/argoverse_h5_files
```

#### TrajNet++
Training AutoBot-Joint on TrajNet++:
```
python train.py --exp-id test --seed 1 --dataset trajnet++ --model-type Autobot-Joint --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/npy_files
```

#### Interaction-Dataset
Training AutoBot-Joint on the Interaction-Dataset while using the raw road segments in the map:
```
python train.py --exp-id test --seed 1 --dataset interaction-dataset --model-type Autobot-Joint --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/interaction_dataset_h5_files
```


### Evaluating an AutoBot model

For all experiments, you can evaluate the trained model on the validation dataset by running:
```
python evaluate.py --dataset-path /path/to/root/of/interaction_dataset_h5_files --models-path results/{Dataset}/{exp_name}/{model_epoch}.pth --batch-size 64
```
Note that the batch-size may need to be reduced for the Interaction-dataset since evaluation is performed on all agent scenes.


### Extra scripts 

We also provide extra scripts that can be used for submitting to the nuScenes, Argoverse and Interaction-Dataset 
Evaluation server.

For nuScenes:

```
python useful_scripts/generate_nuscene_results.py --dataset-path /path/to/root/of/nuscenes_h5_files --models-path results/Nuscenes/{exp_name}/{model_epoch}.pth 
```

For Argoverse:

```
python useful_scripts/generate_argoverse_test.py --dataset-path /path/to/root/of/argoverse_h5_files --models-path results/Argoverse/{exp_name}/{model_epoch}.pth 
```

For the Interaction-Dataset:

```
python useful_scripts/generate_indst_test.py --dataset-path /path/to/root/of/interaction_dataset_h5_files --models-path results/interaction-dataset/{exp_name}/{model_epoch}.pth 
```

## Reference

If you use this repository, please cite our work:

```
@inproceedings{
  girgis2022latent,
  title={Latent Variable Sequential Set Transformers for Joint Multi-Agent Motion Prediction},
  author={Roger Girgis and Florian Golemo and Felipe Codevilla and Martin Weiss and Jim Aldon D'Souza and Samira Ebrahimi Kahou and Felix Heide and Christopher Pal},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=Dup_dDqkZC5}
}
```


