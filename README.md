# PseudoNeuronGAN: Unpaired synthetic image to pseudo-neuron image translation for unsupervised neuron instance segmentation in microscopic images of macaque brain


We provide PyTorch implementations for unpaired synthetic image to pseudo-neuron image translation.


## Requirements
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- pytorch 1.13.0
- pytorch-cuda 11.6
- torchvision 0.14.0


## Test
Download the pre-trained weight in ```PseudoNeuronGAN``` directory ([google drive](https://drive.google.com/drive/...)).conda

The model netG_A2B.pth translates from synthetic image to pseudo-neuron image, netG_B2A.pth is used in another direction. We are interested in the A2B direction.

```
python test.py 
``` 

Note: The model directly translates all the images in the directory 'test'.

An example of how a folder can look like. 
```
/test/synthetic/synthetic_ctx_nbimage14_nbneuron_24_bw.png
/test/synthetic/synthetic_DG_nbimage5_nbneuron_225_bw.png
/test/synthetic/synthetic_globus_nbimage24_nbneuron_5_bw.png
...
/test/neuron/CJ1301_slide81unsharp_subiculum_para_R.png
/test/neuron/CJ1301_slide101unsharp_ctx_L_layer2.png
...
```

<img src="https://github.com/zhenzhen89/PseudoNeuronGAN/images/results.png" width="800"/>



## Train

An example of how a folder can look like. 

```
/train/synthetic/synthetic_CA_nbimage1_nbneuron_26_bw.png
/train/synthetic/synthetic_caudate_nbimage1_nbneuron_16_bw.png
/train/synthetic/synthetic_claustrum_nbimage1_nbneuron_54_bw.png
...
/train/centroid_synthetic/synthetic_CA_nbimage1_nbneuron_26_centroid_visual.png
/train/centroid_synthetic/synthetic_caudate_nbimage1_nbneuron_16_centroid_visual.png
/train/centroid_synthetic/synthetic_claustrum_nbimage1_nbneuron_54_centroid_visual.png
...
/train/neuron/CJ1301_slide81unsharp_CA1_R.png
/train/neuron/CJ1301_slide81unsharp_globus_pallidus_int_R.png
/train/neuron/CJ1301_slide101unsharp_DG_R.png
...
```

```
python train.py 
``` 

* `batch_size`: type=int, default=1.
* `image_size`: type=int, default=512.
* `epochs`: type=int, default=100.
* `learning_rate`: type=float, default=1e-5.
* `input_nc` : type=int, default=3.
* `output_nc` : type=int, default=3.


Note: The model directly trains the PseudoNeuronGAN model based the images in the directory 'train'.


## Acknowledgments
Our code is inspired by [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN).
