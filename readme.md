# AstroCLIP

NewAstroCLIP is a novel, cross-modal, self-supervised foundation model that creates a shared embedding space for multi-band imaging and optical spectra of galaxies. These embeddings encode meaningful physical information shared between both modalities and can be used as the basis for competitive zero-shot learning on various downstream tasks, including similarity search and redshift estimation.

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


# Requirements

This repo should only have basic pytorch and huggingface requirements. The following should install all that is needed:

```bash
pip install -r requirements.txt
```
