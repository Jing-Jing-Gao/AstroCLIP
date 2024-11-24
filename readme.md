# AstroCLIP

<a href="https://arxiv.org/abs/2310.03024" style='vertical-align:middle; display:inline;'><img
							src="https://img.shields.io/badge/astro--ph.IM-arXiv%3A2310.03024-B31B1B.svg" class="plain" style="height:25px;" /></a>

Official PyTorch implementation and pre-trained models for the paper **AstroCLIP: A Cross-Modal Foundation Model for Galaxies**.

![image](assets/im_embedding.png)

AstroCLIP is a novel, cross-modal, self-supervised foundation model that creates a shared embedding space for multi-band imaging and optical spectra of galaxies. These embeddings encode meaningful physical information shared between both modalities, and can be used as the basis for competitive zero- and few-shot learning on a variety of downstream tasks, including similarity search, redshift estimation, galaxy property prediction, and morphology classification.

## Web App
Check out our interactive similarity search app, enabling both in-modal and cross-modal search for galaxies:
https://astroclip.streamlit.app/

## Data Access

The AstroCLIP model is trained on the cross-matched sample containing optical spectra from the [Dark Energy Spectroscopic Instrument (DESI)](https://www.desi.lbl.gov/) Early Data Release (EDR) and multi-band images (g,r,z) from the [DESI Legacy Survey](https://www.legacysurvey.org/) prepared by [Stein, et al. (2022)](https://github.com/georgestein/ssl-legacysurvey/tree/main). We provide the dataset as a HuggingFace dataset, which can be accessed directly using

```python
from datasets import load_dataset

# This downloads about 60 GB of data
dset = load_dataset('astroclip/data/dataset.py')
```

For reproducibility, we include the scripts and a brief description of how to generate the cross-matched dataset in `astroclip/data/crossmatch`.

### Image Pretraining Dataset

![image](assets/decals.png)

While the AstroCLIP and Spectrum Encoder models are trained on the image-spectrum dataset, we pretrain the galaxy image model separately on full Stein, et al. (2022) image dataset, which consists of 76M galaxy images. This dataset can be accessed using this globus endpoint:

https://app.globus.org/file-manager?origin_id=9fb0fc0e-e760-11ec-9bd2-2d2219dcc1fa&origin_path=%2F

The directory is organized into south and north surveys, where each survey is split into chunks of 1,000,000 galaxies (sorted by decreasing z-band flux) and saved in hdf5 format. For more details, see [here](https://github.com/georgestein/ssl-legacysurvey/tree/main).

## Pretraining

AstroCLIP is trained using a two-step process:

1. We pre-train a single-modal galaxy image encoder and a single-modal galaxy spectrum encoder separately.
2. We CLIP-align these two encoders on a paired image-spectrum dataset.


## Results

- Both image and spectra encoders can extract meaningful physical information from the input data.
- The embeddings of both images and spectra are well aligned, allowing us to retrieve spectra that correspond to a given image and vice-versa.

## Products: Datasets and Trained Models

### Dataset

The dataset is derived from a combination of DESI Legacy Survey g, r, z images and DESI Early Data Release spectra. These images are a subset of the [ssl-legacysurvey](https://github.com/georgestein/ssl-legacysurvey) sample compiled by @georgestein from the Legacy Survey DR9.

For convenience, a Hugging Face Datasets loading script is provided, which will automatically download the required data and prepare the dataset on your computer.

```python
from datasets import load_dataset

# This downloads about 60 GB of data
dset = load_dataset('astroclip/datasets/legacy_survey.py')

```


### Training scripts and model weights 

A Jupyter notebook, newastroclip.ipynb, is provided to train the model. You can run it with CUDA. Training was performed on an A100 GPU provided by Google Colab, taking approximately 20 minutes per epoch.

Additionally, a Python script newastroclip.py is provided. You can run the pipeline using:

```bash
pip newastroclip.py
```
## Downstream Tasks

We demonstrate that the AstroCLIP can be used to easily perform a variety of downstream tasks. In particular, we demonstrate their ability to do:

1. In-modal and cross-modal similarity search
2. Photometric redshift prediction
3. Physical property estimation from images
4. Physical property estimation from spectra


The details of these downstream tasks and the results in our paper can be found in `astroclip/downstream_tasks`.


# Requirements

This repo should only have basic pytorch and huggingface requirements. The following should install all that is needed:

```bash
pip install -r requirements.txt
```
