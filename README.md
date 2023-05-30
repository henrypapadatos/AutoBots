# Multimodal joint trajectory prediction - Deep learning for Autonomous Vehicles EPFL
This repository is a fork of the [official implementation](https://github.com/roggirg/AutoBots) of the [AutoBots](https://arxiv.org/abs/2104.00563) architectures.

## Contribution Overview

### Optimizer and activation function 
A first potential amelioration is to use the AdamW optimizer instead of the classic Adam one. Opposed to Adam, AdamW does not put the weight decay term in the moving average. This change [often results](https://towardsdatascience.com/why-adamw-matters-736223f31b5d#:~:text=The%20authors%20show%20experimentally%20that,stochastic%20gradient%20descent%20with%20momentum.) in a better training loss and less overfitting.

Another improvement that can be tested is to replace the ReLU activation function with GELU, as GELU's smoother nature allows the model to [better capture](https://www.saltdatalabs.com/blog/deep-learning-101-transformer-activation-functions-explainer-relu-leaky-relu-gelu-elu-selu-softmax-and-more) complex patterns within the data.

### Learned positional encoding
The positional encoding implemented in the AutoBot architecture is the classic sinusoidal positional encoding used by Vaswani et al. in the [first transformer architecture](https://arxiv.org/abs/1706.03762). With _t_ being the position of the input token in the sequence and _d_ being the encoding dimension. The sinusoidal positional encoding $\overrightarrow{p_t}$ can be written as follow:

$$
{\overrightarrow{p_t}}^{(i)}=f(t)^{(i)}:= \begin{cases}\sin \left(\omega_k \cdot t\right), & \text { if } i=2 k \\\ \cos \left(\omega_k \cdot t\right), & \text { if } i=2 k+1\end{cases}
$$

where
$$\omega_k=\frac{1}{10000^{2 k / d}}$$

Here is a vizualisation if the sinusoidal positional encoding as presented in [this](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) blogpost:
<img src="https://github.com/henrypapadatos/AutoBots/assets/63106608/6ca6d358-64fe-42c5-acdb-6602bd59741c"  width="80%" height="40%">

An alternative is to learn the positional encoding as a model parameter, which could provide the model with more flexibility in representing the relationship between time steps. As learned positional encoding is becoming increasingly popular, we decided to test it within the AutoBot architecture.


### Multi-layer loss
In the initial AutoBot implementation, the computation of loss is limited to the final layer of the decoder. However, in our proposed methodology, we aim to calculate the loss across all the decoder layers and perform backpropagation based on the average of these losses. The diagram below provides a clear representation of the multi-layer loss implementation:

![schema](https://github.com/henrypapadatos/AutoBots/assets/63106608/f07a0ffe-6b99-4b03-9929-14a399cc9f21)


By incorporating the gradients at each layer, we facilitate better propagation of loss throughout the model. This approach is expected to enhance the stability of the learning process and potentially improve overall performance.


## Experimental Setup

Due to many challenges faced with SCITAS, we have sought the assistance of a friend who possesses a setup with 2 NVIDIA GeForce RTX 2080 Ti GPUs. For all our experiments, we trained the models across 150 epochs, which translates to approximately 15 hours of training time. To monitor and record the results of all our experiments, we used [Wandb](https://wandb.ai/), a platform for tracking and visualizing machine learning experiments. The complete logs for our experiments can be found at this [link](https://wandb.ai/henrypapadatos/Autobot?workspace=user-henrypapadatos).

As evaluation metrics, we will use the Min ADE which is the average L2 distance between the predicted and ground truth trajectories. And we will also use the
Min FDE which is the averaged L2 distance between the final points of the most likely prediction and ground truth across all agents.
In our opinion, these 2 metrics represent well the quality of the predictions.

## Getting started

### Description of the dataset
Argoverse 1 Motion Forecasting is a compilation of 324’557 scenarios used for training and validation. Each scenario is 5 seconds long and includes the 2D centroid coordinates of tracked objects from a bird’s-eye view perspective. These centroid coordinates are sampled at a rate of 10 Hz.

To create this collection, extensive analysis was performed on over 1000 hours of driving data obtained from self-driving test vehicles. The focus was on identifying and including the most challenging segments, such as those featuring vehicles at intersections, vehicles making left or right turns, and vehicles changing lanes.

Our model incorporates the map segments and the 2D trajectories of all vehicles in the scene as inputs.


### Argoverse Python package - Installation

Here is the information explaining how to install the Argoverse dataset as found in AutoBots/datasets/argoverse/README.md : 

Follow the instructions [here](https://github.com/argoai/argoverse-api) to 
download the dataset files and install the Argoverse API.

After downloading the dataset files, 
extract the contents of each dataset split file such that the final folder 
structure of the dataset looks like this:
```
argoverse
 └──train
      └──Argoverse-Terms_of_Use.txt
      └──data
          └──1.csv
          └──2.csv
          └──...
 └──val
      └──Argoverse-Terms_of_Use.txt
      └──data
          └──1.csv
          └──2.csv
          └──...
 └──test
      └──Argoverse-Terms_of_Use.txt
      └──data
          └──1.csv
          └──2.csv
          └──...
```

Afterwards, run the following to create the h5 files of the dataset:

```
python create_h5_argo.py --raw-dataset-path /path/to/argoverse --split-name [train/val/test] --output-h5-path /path/to/output/h5_files

```
for both train and val.

Time to create and disk space taken:

| Split       | Time        | Final H5 size |
| ----------- | ----------- | ------------- |
| train       | 4 hours     | 4 GB          |
| val         | 1 hour      | 770 MB        |
| test        | 2 hours     | 1 GB          |


### Training an AutoBot model
Training AutoBot-Ego on Argoverse while using the raw road segments in the map:
```
python train.py --exp-id test --seed 1 --dataset Argoverse --model-type Autobot-Ego --num-modes 6 --hidden-size 128 --num-encoder-layers 2 --num-decoder-layers 2 --dropout 0.1 --entropy-weight 40.0 --kl-weight 20.0 --use-FDEADE-aux-loss True --use-map-lanes True --tx-hidden-size 384 --batch-size 64 --learning-rate 0.00075 --learning-rate-sched 10 20 30 40 50 --dataset-path /path/to/root/of/argoverse_h5_files
```
### Evaluating an AutoBot model
For all experiments, you can evaluate the trained model on the validation dataset by running:
```
python inference.py --dataset-path /path/to/root/of/interaction_dataset_h5_files --models-path results/{Dataset}/{exp_name}/{model_epoch}.pth --batch-size 64
```
You can download the weights of the best model by clicking on [this link](https://drive.google.com/file/d/1Uiu67p2FoDJu8p6ymUzBbwCsTgXs-dLb/view?usp=sharing). 
And you can then run the inference script by executing the following line: 
```
python inference.py --dataset-path /path/to/root/of/interaction_dataset_h5_files --models-path best_model.pth --activation_function GELU --batch-size 64
```

## Results
### Optimizer and activation function 
To test these experiments, use the following arguments:
```
--activation_function GELU or ReLU
```
```
--optimizer AdamW or Adam
```
|              | Val minADE 5 | Val minADE 6 | Val minFDE 6 |
|--------------|--------------|--------------|--------------|
| AdamW +ReLU       | **0.7085**       | 0.6528       | 1.112        |
| AdamW + GELU | 0.7093       | **0.651**        | **1.101**        |
| Autobots (Adam +ReLU)     | 0.7159       | 0.6562       | 1.112        |

The Autobots architecture is our baseline, it is implemented with Adam and ReLU. 
We observe that AdamW + ReLU performs better for the minADE 5 metric and AdamW + GELU performs better for minADE 6 and minFDE 6. However, these amelioration are not large. 

### Learned positional encoding
To test this experiment, use the following argument: 
´´´
--positional_embedding standard or learned
´´´

|            | Val minADE 5 | Val minADE 6 | Val minFDE 6 |
|------------|--------------|--------------|--------------|
| Learned PE | 0.7531       | 1.155        | **0.6774**       |
| Autobots (Sinusoidal PE)  | **0.7159**       | **0.6562**       | 1.112        |

The results of this experiment are intriguing. The learned PE drastically improve the performances of Val MinADE 6 but reduces them by a lot for minFDE 6. In light of these results, we decided to stick to the classic sinusoidal PE for the following experiments.

### Multi-layer loss (MLL)
To test this experiment, use the following arguments: 
```
--num-decoder-layers
```
```
--multi_stage_loss True or False 
```

Along with testing the multilayer loss, we also tried to add more decoder layers. 

|               | Val minADE 5 | Val minADE 6 | Val minFDE 6 |
|---------------|--------------|--------------|--------------|
| num-decoder-layers 3       | 0.7169       | **0.6524**       | **1.105**        |
| num-decoder-layers 4       | 0.7205       | 0.6563       | 1.111        |
| num-decoder-layers 3 & MLL | 0.734        | 0.6572       | 1.118        |
| num-decoder-layers & MLL | 0.7382       | 0.6527       | 1.106        |
| Autobots  (num-decoder-layers 2)  | **0.7159**       | 0.6562       | 1.112        |

It appears that increasing the number of decoders does not significantly enhance performance, except for slight improvements in Val minFDE 6. Interestingly, the metric curves from the test sets demonstrate better results with a larger architecture, as illustrated in the graph below.
<img src="https://github.com/henrypapadatos/AutoBots/assets/63106608/6d8a7901-3ecf-45a8-81b0-79c689ba6280"  width="80%" height="40%">
This discrepancy suggests the presence of overfitting. To address this issue, we have opted to experiment with varying levels of L2 regularization in our final experiment.

To test this experiment, use the following argument: 
```
--L2_regularization
```

|                       | Val minADE 5 | Val minADE 6 | Val minFDE 6 |
|-----------------------|--------------|--------------|--------------|
| num-decoder-layers 4 & MLL & AdamW & L2 regularization 0.025 & GELU          |   0.716           |       **0.6494**      |     **1.104**         |
| num-decoder-layers 4 & MLL & AdamW & L2 regularization 0.015 & GELU         |      0.7152        |          0.6536    |     1.12         |
| num-decoder-layers 4 & MLL & AdamW & L2 regularization 0.01 | 0.7252       | 0.6536       | 1.113        |
| AdamW (num-decoder-layers 2 & AdamW & L2 regularization 0.01)                | **0.7085**       | 0.6528       | 1.112        |
| Autobots  (num-decoder-layers 2 & Adam & L2 regularization 0.01)            | 0.7159       | 0.6562       | 1.112        |

We observe that increasing the L2 regularization led to small improvement of the Val minADE 6 and the Val minFDE 6. Hovewer, these amelioration are not consequent. Further work could test the impact of increasing this parameter even more or one could try other regularization techniques. 


## Challenges Faced and Solutions
This section highlights the obstacles we encountered throughout our project and the solutions we adopted. Had this section reflected the time spent, it would have been the most extensive.  However, let’s keep it short!

1. **Permission issues and long waiting time on SCITAS:** To resolve this, we decided to use our friend's GPUs instead.

2. **Setting up custom VPN and SSH connection for GPU access:** As it was our first time configuring such connections, we faced multiple challenges. However, after several attempts, we managed to set it up successfully.

3. **Argoverse dataset installation and creation of the h5 file:** We faced difficulty unzipping the dataset on the virtual machine. So, we unzipped it on our laptop using 7zip and used `scp` to transfer the dataset, which took approximately 15 hours.

4. **Incorporating SSL pretraining tasks in the Autobot training procedure:** Initially, we aimed to include pretraining tasks from [SSL](https://github.com/AutoVision-cloud/SSL-Lanes) into Autobot's training procedure. However, we encountered several issues with the Horovod package installation, which is used for multi-GPU training. Despite multiple attempts and even trying to rewrite the code without Horovod, we couldn't successfully implement it.

Considering the deadline, we shifted our focus and decided to work on modifying Autobot without including the SSL pretraining step. Consequently, we carried out the experiments described in the previous sections.

## Conclusion

In conclusion, we have encountered the challenges of setting up a deep learning environment. Ensuring that all the required dependencies and datasets are properly installed on a virtual machine proved to be a complex task, and we now have a deeper appreciation for this process. We are happy to have experimented with various modifications to the Autobot architecture and make incremental improvements. Our best-performing model was trained using the following parameters:
- Optimizer: AdamW
- Activation function: GELU
- L2 regularization: 0.025
- Type of loss: Multi-Layer loss
- Number of decoder layers: 4

## Reference

Girgis, R. (2022, January 28). Latent Variable Sequential Set Transformers for Joint Multi-Agent Motion Prediction. OpenReview. https://openreview.net/forum?id=Dup_dDqkZC5


