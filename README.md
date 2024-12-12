# Estimating Barycenters of Distributions with Neural Optimal Transport
 
This is the official `Python` implementation of the [ICML 2024](https://icml.cc/virtual/2024/poster/32654) paper **Estimating Barycenters of Distributions with Neural Optimal Transport** by [Alexander Kolesov](https://scholar.google.com/citations?user=vX2pmScAAAAJ&hl=ru&oi=ao), [Petr Mokrov](https://scholar.google.com/citations?user=CRsi4IkAAAAJ&hl=ru&oi=sra), [Igor Udovichenko](https://scholar.google.com/citations?hl=ru&user=IkcYVhXfeQcC), [Milena Gazdieva](https://scholar.google.com/citations?user=h52_Zx8AAAAJ&hl=ru&oi=sra), [Gudmund Pammer](https://scholar.google.com/citations?user=ipItetYAAAAJ&hl=ru&oi=sra), [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru) and [Alexander Korotin](https://scholar.google.com/citations?user=1rIIvjAAAAAJ&hl=ru&oi=sra).

<p  align="center">
  <img src= "pics/maxminbary_teaser.png" width="700" />
</p>

## Pre-requisites
The implementation is GPU-based. Single GPU GTX 1080 ti is enough to run each particular experiment. We tested the code with `torch==2.1.1+cu121`. The code might not run as intended in older/newer `torch` versions. Versions of other libraries are specified in `requirements.txt`. Pre-trained models for maps and potentials are located [here](https://disk.yandex.ru/d/GAhMTWmB4KEVmA).

 
## Repository structure

All the experiments are issued in the form of pretty self-explanatory jupyter notebooks ( `stylegan2/notebooks/ `).

- `src/` - auxiliary source code for the experiments: training, plotting, logging, etc.
- `stylegan2/` - folder with auxiliary code for using StyleGAN2.
- `stylegan2/notebooks` - jupyter notebooks with evaluation of barycenters on 2D and Image datasets.
- `data/` - folder with datasets. 
- `SG2_ckpt/` - folder with checkpoints for [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) models.
 
### 2-Dimensional estimating barycenters

- `stylegan2/notebooks/twister2D.ipynb` -- toy experiments on 2D Twister dataset;
- `stylegan2/notebooks/Gauss2D.ipynb` -- evaluating metrics of our method in Gaussian case.

<p align="center"><img src="pics/twister.png" width="700" /></p>

### High-Dimensional estimating barycenters of Ave,Celeba! dataset

- `stylegan2/notebooks/AVE_CELEBA_L2.ipynb` -- estimating barycenters of **Ave,Celeba** dataset in Image space ;
- `stylegan2/notebooks/AVE_CELEBA_LATENT.ipynb` --  estimating barycenters in latent space with classical cost;
- `stylegan2/notebooks/AVE_CELEBA_ENTROPY.ipynb` -- estimating barycenters in latent space with $\epsilon$-KL cost ;
- `stylegan2/notebooks/AVE_CELEBA_KERNEL.ipynb` --  estimating barycenters in latent space with $\gamma$-Energy cost;
- `stylegan2/notebooks/AVE_CELEBA_KERNEL_GAUSS.ipynb` -- estimating barycenters in latent space with $\gamma$-Energy cost(Gaussian reparam.).

<p  align="center">
  <img src= "pics/ave_0.png" width="260" />
  <img src="pics/ave_1.png" width="230" />  
  <img src="pics/ave_2.png" width="230" /> 
</p>

### High-Dimensional estimating barycenters of Colored MNIST dataset

- `stylegan2/notebooks/SHAPE_COLOR_EXPERIMENT_ENTROPIC.ipynb` -- estimating barycenters in latent space with $\epsilon$-KL cost.

<p  align="center">
  <img src= "pics/shape_shape_1.png" width="340" />
  <img src="pics/color_color_1.png" width="340" /> 
</p>

### Estimating barycenters of MNIST dataset, digits 0 and 1

- `stylegan2/notebooks/MNIST_01_L2_DATA.ipynb` -- estimating barycenters in **data** space (L2 cost);
- `stylegan2/notebooks/MNIST_01_KERNEL_DATA.ipynb` -- estimating barycenters in **data** space with $\gamma$-Energy cost.

<p  align="center">
  <img src= "pics/mnist_01_data_energyreg.png" width="230" />
</p>


## How to Use

- Download the repository.

```console
git clone https://github.com/justkolesov/NOTBarycenters.git
```
- Create virtual environment 

```console
pip install -r requirements.txt
```
- Download either [MNIST](https://yann.lecun.com/exdb/mnist) or [Ave, Celeba!](https://disk.yandex.ru/d/3jdMxB789v936Q) 64x64 dataset.

- Set downloaded dataset in appropriate subfolder in `data/`.

- If you run High-dimensional Image experiment, download appropriate [StyleGan2](https://github.com/NVlabs/stylegan2-ada-pytorch) model from [here](https://disk.yandex.ru/d/GAhMTWmB4KEVmA) (folder `StyleGan2/`).

- Set downloaded StyleGan2 model in appropriate subfolder in `SG2_ckpt/`.

- Run notebook for training or take appropriate checkpoint from [here](https://disk.yandex.ru/d/GAhMTWmB4KEVmA) and upload them.

## Credits

- [Ave,Celeba](https://disk.yandex.ru/d/3jdMxB789v936Q) with faces dataset;
- [MNIST](https://yann.lecun.com/exdb/mnist) with images dataset.
- [UNet architecture](https://github.com/milesial/Pytorch-UNet) for maps in Image spaces;
- [ResNet architectures](https://github.com/harryliew/WGAN-QC) for maps in latent spaces;
