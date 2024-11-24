import os
import spender
from datasets import load_dataset, DatasetDict
import torch
import torch.nn as nn
import lightning as L
from pl_bolts.models.self_supervised import Moco_v2
import numpy as np
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomErasing, ToTensor, CenterCrop, ToPILImage
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import binned_statistic_2d
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from torch import tensor
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from sklearn.metrics import r2_score
from datasets import load_dataset, DatasetDict
import os
import lightning as L
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d



google_drive_path = '/content/drive/My Drive/datasets_astroclip'
dataset = load_dataset('/content/drive/MyDrive/legacy_survey.py', cache_dir=google_drive_path)
dataset.set_format(type='torch', columns=['image', 'spectrum', 'redshift', 'targetid'])
train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=512, shuffle=True, num_workers=10)
val_dataloader = torch.utils.data.DataLoader(dataset['test'], batch_size=512, shuffle=False, num_workers=10)

# Define Transforms to be used during training
image_transforms = Compose([
        # ToRGB(),
        # ToTensor(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        CenterCrop(96),
])


def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):
    import numpy as np
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    rgb = np.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

def scatter_plot_as_images(z_emb, images, nx=8, ny=8, npix_show=96, iseed=13579, display_image=True):
    """Sample points from scatter plot and display as their original galaxy image

    Parameters
    ----------
    DDL : class instance
        DecalsDataLoader class instance
    z_emb: array
        (N, 2) array of the galaxies location in some compressed space.
        If second axis has a dimensionality greater than 2 we only consider the leading two components.
    """
    z_emb = z_emb[:, :2] # keep only first two dimensions

    nplt = nx*ny

    img_full = np.zeros((ny*npix_show, nx*npix_show, 3)) + 255#, dtype=np.uint8) + 255

    xmin = z_emb[:,0].min()
    xmax = z_emb[:,0].max()
    ymin = z_emb[:,1].min()
    ymax = z_emb[:,1].max()

    dz_emb = 0.25
    dx_cent = z_emb[:,0].mean()
    dy_cent = z_emb[:,1].mean()

    dx_cent = 10.0
    dy_cent = 7.0


    binx = np.linspace(xmin,xmax, nx+1)
    biny = np.linspace(ymin,ymax, ny+1)

    ret = binned_statistic_2d(z_emb[:,0], z_emb[:,1], z_emb[:,1], 'count', bins=[binx, biny], expand_binnumbers=True)
    z_emb_bins = ret.binnumber.T

    inds_used = []
    inds_lin = np.arange(z_emb.shape[0])

    # First get all indexes that will be used
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:,0]==ix) & (z_emb_bins[:,1]==iy)
            inds = inds_lin[dm]

            np.random.seed(ix*nx+iy+iseed)
            if len(inds) > 0:
                ind_plt = np.random.choice(inds)
                inds_used.append(ind_plt)# inds_use[ind_plt])

    # load in all images
    iimg = 0

    # Add each image as postage stamp in desired region
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:,0] == ix) & (z_emb_bins[:,1]==iy)
            inds = inds_lin[dm]

            np.random.seed(ix*nx+iy+iseed)
            if len(inds) > 0:

                # imi = images[inds[0]][28:-28, 28:-28]
                # imi = images[inds[0]]
                imi = dr2_rgb(images[inds[0]].T, bands=['g', 'r', 'z'])
                img_full[iy*npix_show:(iy+1)*npix_show, ix*npix_show:(ix+1)*npix_show] = imi

                iimg += 1

    if display_image:
        plt.figure(figsize=(nx, ny))
        plt.imshow(img_full, origin='lower')#, interpolation='none')
        plt.savefig("/content/drive/MyDrive/pca_after.png")
        plt.axis('off')

    return img_full

moco_model = Moco_v2.load_from_checkpoint(checkpoint_path='resnet50.ckpt')
class OutputExtractor(L.LightningModule):
    """
    Pass data through network to extract model outputs
    """
    def __init__(self, backbone: torch.nn.Module):
        super(OutputExtractor, self).__init__()
        self.backbone = backbone
        self.backbone.train()

    def forward(self, batch):
        x, _ = batch
        z_emb = self.backbone(x)
        return z_emb

    def predict(self, batch, batch_idx: int, dataloader_idx: int=None):
        return self(batch)

# image encoder
backbone = moco_model.encoder_q
img_model = OutputExtractor(backbone).to('cuda')
# img_model = OutputExtractor(backbone)
num_params = np.sum(np.fromiter((p.numel() for p in img_model.parameters()), int))
print(f"Number of parameters in image model: {num_params:,}")

# spectrum enoder
spender.hub.list()
sdss, spender_model = spender.hub.load('sdss_II')


class MLP(nn.Sequential):
    """Multi-Layer Perceptron

    A simple implementation with a configurable number of hidden layers and
    activation functions.

    Parameters
    ----------
    n_in: int
        Input dimension
    n_out: int
        Output dimension
    n_hidden: list of int
        Dimensions for every hidden layer
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):

        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 1):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))

        super(MLP, self).__init__(*layer)


class SpectrumEncoder(nn.Module):
    """Spectrum encoder

    Modified version of the encoder by Serrà et al. (2018), which combines a 3 layer CNN
    with a dot-product attention module. This encoder adds a MLP to further compress the
    attended values into a low-dimensional latent space.

    Paper: Serrà et al., https://arxiv.org/abs/1805.03908

    Parameters
    ----------
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """

    def __init__(
        self, instrument, n_latent, n_hidden=(128, 64, 32), act=None, dropout=0
    ):

        super(SpectrumEncoder, self).__init__()
        self.instrument = instrument
        self.n_latent = n_latent

        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(
            filters, sizes, dropout=dropout
        )
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(
            nn.MaxPool1d(s, padding=s // 2) for s in sizes[:2]
        )
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features to latents
        if act is None:
            act = [nn.PReLU(n) for n in n_hidden]
            # last activation identity to have latents centered around 0
            act.append(nn.Identity())
        self.mlp = MLP(self.n_feature, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout)

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i - 1]
            f = filters[i]
            s = sizes[i]
            p = s // 2
            conv = nn.Conv1d(
                in_channels=f_in,
                out_channels=f,
                kernel_size=s,
                padding=p,
            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def _downsample(self, x):
        # compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1] // 2
        # split half channels into attention value and key
        h, a = torch.split(x, [C, C], dim=1)

        return h, a

    def forward(self, y):
        """Forward method

        Parameters
        ----------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra

        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        # run through CNNs
        h, a = self._downsample(y)
        # softmax attention
        a = self.softmax(a)

        # attach hook to extract backward gradient of a scalar prediction
        # for Grad-FAM (Feature Activation Map)
        if ~self.training and a.requires_grad == True:
            a.register_hook(self._attention_hook)

        # apply attention
        x = torch.sum(h * a, dim=2)

        # run attended features into MLP for final latents
        x = self.mlp(x)
        return x

    @property
    def n_parameters(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _attention_hook(self, grad):
        self._attention_grad = grad

    @property
    def attention_grad(self):
        """Gradient of the attention weights

        Factor to compute the importance of attention for Grad-FAM method.

        Requires a previous `loss.backward` call for any scalar loss function based on
        outputs of this class's `forward` method. This functionality is switched off
        during training.
        """
        if hasattr(self, "_attention_grad"):
            return self._attention_grad
        else:
            return None
        

class CLIPLoss(nn.Module):
    def get_logits(self, image_features, spectrum_features, logit_scale):
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(self, image_features, spectrum_features, logit_scale, output_dict=False):
        logits_per_image, logits_per_spectrum = self.get_logits(image_features, spectrum_features, logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_spectrum, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss
    

# Define image and spectrum encoders
image_encoder = OutputExtractor(backbone).to('cuda')
# spender_model = spender_model.to("cuda")
spectrum_encoder = SpectrumEncoder(None, 128).to("cuda")

num_params_im = sum(p.numel() for p in image_encoder.parameters())
num_params_sp = sum(p.numel() for p in spectrum_encoder.parameters())
print(f"Image Encoder Params: {round(num_params_im/1e6, 1)} M, Spectrum Encoder Params: {round(num_params_sp/1e6, 1)} M")

class AstroCLIP(L.LightningModule):
    def __init__(self, image_encoder, spectrum_encoder):
        super().__init__()
        self.image_encoder = image_encoder

        # Freeze all but the last layers of the image encoder
        for name, child in self.image_encoder.backbone.named_children():
            if name != 'fc':
                for param in child.parameters():
                    param.requires_grad = False

        # Spectrum encoder is already frozen, so simply load it in
        self.spectrum_encoder = spectrum_encoder

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(15.5), requires_grad=False)
        self.criterion = CLIPLoss()

        # Initialize accumulators and counters for storing loss values
        self.train_loss_sum = 0
        self.val_loss_sum = 0
        self.train_batches = 0
        self.val_batches = 0
        self.train_losses = []
        self.val_losses = []

    def forward(self, x, image=True):
        if image:
            # Embed image
            embedding = self.image_encoder((x,None))
        else:
            # Embed spectrum
            embedding = self.spectrum_encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        im, sp = batch['image'].transpose(1, 3), batch['spectrum'].squeeze()
        im = image_transforms(im)
        image_features = self.image_encoder((im, None))
        spectrum_features = self.spectrum_encoder(sp)

        loss_withlogit = self.criterion(image_features, spectrum_features, 15.5)
        loss_nologit = self.criterion(image_features, spectrum_features, 1)

        # Accumulate loss and batch count
        self.train_loss_sum += loss_withlogit.item()
        self.train_batches += 1

        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        im, sp = batch['image'].transpose(1, 3), batch['spectrum'].squeeze()
        im = image_transforms(im)
        image_features = self.image_encoder((im, None))
        spectrum_features = self.spectrum_encoder(sp)
        val_loss_withlogit = self.criterion(image_features, spectrum_features, 15.5)

        # Accumulate loss and batch count
        self.val_loss_sum += val_loss_withlogit.item()
        self.val_batches += 1

    def on_train_epoch_end(self):
        # Compute average training loss
        avg_train_loss = self.train_loss_sum / self.train_batches
        self.train_losses.append(avg_train_loss)  # Store the loss value
        print(f"Epoch {self.current_epoch}: Avg Train Loss = {avg_train_loss:.4f}")

        # Save the model checkpoint

        model_save_path = f"/content/drive/My Drive/after_AstroCLIP_epoch_{self.current_epoch}.pt"
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Reset accumulators and batch count
        self.train_loss_sum = 0
        self.train_batches = 0

    def on_validation_epoch_end(self):
        # Compute average validation loss
        avg_val_loss = self.val_loss_sum / self.val_batches
        self.val_losses.append(avg_val_loss)  # Store the loss value
        print(f"Epoch {self.current_epoch}: Avg Val Loss = {avg_val_loss:.4f}")

        # Reset accumulators and batch count
        self.val_loss_sum = 0
        self.val_batches = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=0.2)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=5e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

# Instantiate the model (ensure image_encoder and spectrum_encoder are defined)
CLIP = AstroCLIP(image_encoder, spectrum_encoder)

# Define the trainer and start training
trainer = L.Trainer(
    max_epochs=20,
    log_every_n_steps=1,
)

# Ensure train_dataloader and val_dataloader are defined
trainer.fit(model=CLIP, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Plot the losses after training
plt.figure(figsize=(10, 5))
plt.plot(CLIP.train_losses, label='Training Loss')
plt.plot(CLIP.val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("loss_curve_after.png")
plt.show()

import torch
import numpy as np
from tqdm import tqdm


dataset.set_format(type='torch', columns=['image', 'spectrum', 'redshift'])
testdata = torch.utils.data.DataLoader(dataset['test'], batch_size=512, shuffle=False, num_workers=10)

from tqdm import tqdm
image_features = []
spectrum_features = []
images = []
spectra = []
redshifts = []
for batch_test in tqdm(testdata):
    # image_single_original = batch_test['image'].clone()
    image_single_original = batch_test['image'].T
    # print(image_single_original.shape)
    image_single_original = image_single_original.permute(3, 0, 1, 2)
    # print(image_single_original.shape)
    image_single_original = image_transforms(image_single_original)
    # print(image_single_original.shape)
    image_single = batch_test['image'].clone().transpose(1,3)
    image_single = image_transforms(image_single)
    images.append(np.stack(image_single_original,axis=0))
    # images.append(image_single)
    # print(image_single.shape)
    image_feature_single = CLIP(image_single.to("cuda")).cpu().detach().numpy()
    image_features.append(image_feature_single)
    CLIP.spectrum_encoder = CLIP.spectrum_encoder.to("cuda")
    spectrum_single = batch_test['spectrum'].clone().squeeze().to("cuda")
    spectrum_single_cpu = batch_test['spectrum'].clone().squeeze()
    spectra.append(np.stack(spectrum_single_cpu,axis=0))
    spectrum_feature_single = CLIP(spectrum_single, image = False).cpu().detach().numpy()
    spectrum_features.append(spectrum_feature_single)
    redshifts.append(np.stack(batch_test['redshift'],axis=0))


image_features_try = np.array(image_features[0:77]).reshape(-1, 128)
im_pca = PCA(n_components=4).fit_transform(image_features_try/np.linalg.norm(image_features_try, axis=-1, keepdims=True))
# print(im_pca.shape)
# Convert the list to a numpy array
# print(np.array(images[0:77]).shape)
images_numpy = np.array(images[0:77]).reshape(-1, 3, 96, 96)
images_numpy = np.transpose(images_numpy, (0, 2, 3, 1))
# print(images_numpy.shape)
images_numpy[0][28:-28, 28:-28]
# .transpose(0, 1, 2, 0)
# print(images_numpy[0].shape)
# Convert the list to a numpy array
numpy_array = np.array(images[0:77]).reshape(-1, 3, 96, 96)


# Use np.squeeze() to remove the first dimension
numpy_array = np.squeeze(numpy_array)
images_new = numpy_array
images_new = torch.tensor(images_new)
image_tensor = images_new[2].permute(1, 2, 0)

spectra_numpy = np.array(spectra[0:77]).reshape(-1, 7781)
l = np.linspace(3586.7408577, 10372.89543574, spectra_numpy[0:1].shape[1])
ind_query = 2

spectra_squeeze = np.array(spectra[0:77]).reshape(-1, 7781)
spectrum_features_try = np.array(spectrum_features[0:77]).reshape(-1, 128)
image_features_try = np.array(image_features[0:77]).reshape(-1, 128)

# Compute the similarities across all of the other objects (for in-modality and cross-modality search)
image_features_normed = image_features_try / np.linalg.norm(image_features_try, axis=-1, keepdims=True)
spectrum_features_normed = spectrum_features_try / np.linalg.norm(spectrum_features_try, axis=-1, keepdims=True)

spectral_similarity = spectrum_features_normed[ind_query] @ spectrum_features_normed.T
image_similarity = image_features_normed[ind_query] @ image_features_normed.T
cross_image_similarity = image_features_normed[ind_query] @ spectrum_features_normed.T
cross_spectral_similarity = spectrum_features_normed[ind_query] @ image_features_normed.T

s = spectral_similarity
inds = np.argsort(s)[::-1]

# Create a single figure with 4 subplots
fig, axs = plt.subplots(4, 2, figsize=(15, 20))

for i in range(4):
    # Plot the image
    axs[i, 0].imshow(dr2_rgb(images_new[inds[i]].permute(1, 2, 0).T, bands=['g', 'r', 'z']))
    if i == 0:
        axs[i, 0].set_title('Retrieved Image')

    # Plot the spectra
    axs[i, 1].plot(l, spectra_squeeze[inds[i]], color='grey', alpha=0.5)
    axs[i, 1].plot(l, gaussian_filter1d(spectra_squeeze[inds[i]], 2), alpha=0.5, label='Spectrum of retrieved object')
    axs[i, 1].plot(l, gaussian_filter1d(spectra_squeeze[ind_query], 2), alpha=0.5, label='Spectrum of query object')
    axs[i, 1].set_ylim(0, 20)  # Adjust y-axis limits as needed
    if i == 0:
        axs[i, 1].set_title('Retrieved Spectrum')
        axs[i, 1].legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/spectral_after_combined.png")
plt.show()

s = image_similarity
inds = np.argsort(s)[::-1]
for i in range(4):
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    plt.imshow(dr2_rgb(images_new[inds[i]].permute(1, 2, 0).T, bands=['g', 'r', 'z']))
    # plt.imshow(images_new[inds[i]].permute(1, 2, 0))
    if i == 0:
        plt.title('Retrieved Image')
    plt.subplot(122)
    plt.plot(l,spectra_squeeze[inds[i]], color='grey', alpha=0.5)
    plt.plot(l,gaussian_filter1d(spectra_squeeze[inds[i]],2),alpha=0.5, label='spectrum of retrieved object')
    plt.plot(l,gaussian_filter1d(spectra_squeeze[ind_query],2),alpha=0.5, label='Spectrum of query object')
    plt.ylim(-0,20)
    if i == 0:
        plt.title('Retrieved Spectrum')
    plt.legend()
plt.savefig("/content/drive/MyDrive/image_after.png")

s = cross_image_similarity
inds = np.argsort(s)[::-1]
for i in range(4):
    print(s[inds])
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    plt.imshow(dr2_rgb(images_new[inds[i]].permute(1, 2, 0).T, bands=['g', 'r', 'z']))
    if i == 0:
        plt.title('Retrieved Image')
    plt.subplot(122)
    plt.plot(l,spectra_squeeze[inds[i]], color='grey', alpha=0.5)
    plt.plot(l,gaussian_filter1d(spectra_squeeze[inds[i]],2),alpha=0.5, label='spectrum of retrieved object')
    plt.plot(l,gaussian_filter1d(spectra_squeeze[ind_query],2),alpha=0.5, label='Spectrum of query object')
    plt.ylim(-0,20)
    if i == 0:
        plt.title('Retrieved Spectrum')
    plt.savefig("/content/drive/MyDrive/image_spectrum_after.png")
    plt.legend()

s = cross_spectral_similarity
inds = np.argsort(s)[::-1]
for i in range(4):
    print(s[inds])
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    plt.imshow(dr2_rgb(images_new[inds[i]].permute(1, 2, 0).T, bands=['g', 'r', 'z']))
    if i == 0:
        plt.title('Retrieved Image')
    plt.subplot(122)
    plt.plot(l,spectra_squeeze[inds[i]], color='grey', alpha=0.5)
    plt.plot(l,gaussian_filter1d(spectra_squeeze[inds[i]],2),alpha=0.5, label='spectrum of retrieved object')
    plt.plot(l,gaussian_filter1d(spectra_squeeze[ind_query],2),alpha=0.5, label='Spectrum of query object')
    plt.ylim(-0,20)
    if i == 0:
        plt.title('Retrieved Spectrum')
    plt.savefig("/content/drive/MyDrive/spectrum_image_after.png")
    plt.legend()

Z_HP = np.array(redshifts[0:77]).reshape(-1)
neigh = KNeighborsRegressor(weights='distance', n_neighbors=16)
neigh.fit(image_features_try[:-5000], Z_HP[:-5000])
preds = neigh.predict(image_features_try[-5000:])

sns.scatterplot(x=Z_HP[-5000:], y=preds, s=5, color=".15")
sns.histplot(x=Z_HP[-5000:], y=preds, bins=64, pthresh=.1, cmap="mako")
sns.kdeplot(x=Z_HP[-5000:], y=preds, levels=5, color="w", linewidths=1)
plt.xlabel('True Redshift')
plt.ylabel('Predicted Redshift')
plt.plot([0,1], [0,1], color='gray', ls='--')
plt.xlim(0,0.65)
plt.ylim(0,0.65)
plt.text(0.05, 0.55, '$R^2$ score: %0.2f'%(r2_score(Z_HP[-5000:], preds)), fontsize='large' )
plt.savefig("/content/drive/MyDrive/image_redshift_after.png")

neigh = KNeighborsRegressor(weights='distance', n_neighbors=16)
neigh.fit(spectrum_features_try[:-5000], Z_HP[:-5000])
preds = neigh.predict(spectrum_features_try[-5000:])

sns.scatterplot(x = Z_HP[-5000:], y=preds, s=5, color=".15")
sns.histplot(x=Z_HP[-5000:], y=preds, bins=64, pthresh=.1, cmap="mako")
sns.kdeplot(x=Z_HP[-5000:], y=preds, levels=5, color="w", linewidths=1)
plt.xlabel('True Redshift')
plt.ylabel('Predicted Redshift')
plt.plot([0,1], [0,1], color='gray', ls='--')
plt.xlim(0,0.65)
plt.ylim(0,0.65)
plt.text(0.05, 0.55, '$R^2$ score: %0.2f'%(r2_score(Z_HP[-5000:], preds)), fontsize='large' )
plt.savefig("/content/drive/MyDrive/spectra_redshift_after.png")
