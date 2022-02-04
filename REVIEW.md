# CartoonGAN

Paper: [CartoonGAN: Generative Adversarial Networks for Photo Cartoonization](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)

Official Github (Lua Torch Version): https://github.com/FlyingGoblin/CartoonGAN

Github (PyTorch Version): https://github.com/znxlwm/pytorch-CartoonGAN

Github (Tensorflow Version): https://github.com/FilipAndersson245/cartoon-gan


"From the perspective of computer vision algorithms, the goal of cartoon stylization is to map images in the photo manifold into the cartoon manifold while keeping the content unchanged."

The paper, as its name suggested, is to perform Image Cartoonization. The paper mentioned the properties of cartoon:
* (1) cartoon styles have unique characteristics with high level simplification and abstraction.
    *  cartoon images are highly simplified and abstracted from real-world photos. It does not equal to apply textures such as brush strokes in many other styles.
* (2) cartoon images tend to have clear edges, smooth color shading and relatively simple textures, which exhibit significant challenges for texture-descriptor-based loss functions used in existing methods.

Key features of CartoonGAN:
* Requires Unpaired images for training
* Produce high-quality cartoon stylization (compare to CycleGAN)
* Less training time than CycleGAN because CartoonGAN only uses 1 generator and 1 discriminator
* A different adversarial loss due to the involvement of edge-smoothed dataset
* A new initialization phase to improve the convergence of the network (Pre-train the generator network with only content loss)

# Loss functions of CartoonGAN

## Edge-promoting Adversarial loss of CartoonGAN

The paper found the training of Discriminator is not sufficient if we simply put True Cartoon images and Generated Cartoon images.

> "we observe that simply training the discriminator
$D$ to separate generated and true cartoon images is not sufficient for transforming photos to cartoons. This is because
the presentation of clear edges is an important characteristic of cartoon images, but the proportion of these edges is
usually very small in the whole image. Therefore, an output image without clearly reproduced edges but with correct
shading is likely to confuse the discriminator trained with a
standard loss."

Since the Cartoon images have clear edges, the Discriminator $D$ has to be focus on the edges and able to classify fake cartoon without edges (even with correct shading). The Generator $G$ has to be guided to convert the input into the correct manifold. Thus the paper proposed a method to create a edge-smoothed version of the original cartoon image dataset as a guidance. The edge-smoothed version dataset is get by applying:

* (1) detect edge pixels using a standard Canny edge detector
* (2) dilate the edge regions
* (3) apply a Gaussian smoothing in the dilated edge regions

Here is the implementation of edge-smoothing:
```python
import cv2
import numpy as np

# Paper author used MedianBlur instead of Gaussian blur: https://github.com/FlyingGoblin/CartoonGAN/issues/11

def edge_smooth(img_path):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    bgr_img = cv2.imread(img_path)
    bgr_img = cv2.resize(bgr_img, (256, 256))
    pad_img = np.pad(bgr_img, ((2,2), (2,2), (0,0)), mode='reflect')

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # detect edge pixels using a standard Canny edge detector
    edges = cv2.Canny(gray_img, 100, 200)
    # dilate the edge regions
    dilation = cv2.dilate(edges, kernel)

    # apply a Gaussian smoothing in the dilated edge regions
    gauss_img = np.copy(bgr_img)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

    # Concatenate 2 images
    result = np.concatenate((bgr_img, gauss_img), 1)

    return result
```

With the new dataset, it can be used to help the Discriminator to learn.

The goal of training the discriminator $D$ is to maximize the probability of assigning the correct label to fake generated cartoon image, the edge-smoothed (without clear edges) version of cartoon images, and the real cartoon images. Thus the Generator $G$ can be guided to convert the input into the correct manifold.

Therefore the edge-promoting adversarial loss is formulaed as:
$$
\begin{aligned}
\mathcal{L}_{a d v}(G, D) &=\mathbb{E}_{c_{i} \sim S_{\text {data }}(c)}\left[\log D\left(c_{i}\right)\right] \\
&+\mathbb{E}_{e_{j} \sim S_{\text {data }}(e)}\left[\log \left(1-D\left(e_{j}\right)\right)\right] \\
&+\mathbb{E}_{p_{k} \sim S_{\text {data }}(p)}\left[\log \left(1-D\left(G\left(p_{k}\right)\right)\right)\right]
\end{aligned}
$$

Where:
* $c_{i} \in S_{d a t a}(c)$ is a real cartoon image.
* $e_{j} \in S_{d a t a}(e)$ is a edge-smoothed cartoon image.
* $p_{k} \in S_{d a t a}(p)$ is a photo.
* $\ell_{1}$ sparse regularization is used (paper stated it is able to cope with such changes much better than the standard ℓ2 norm).
* $G(p_{k})$ is a fake cartoon image that took a photo as input.


## Content loss of CartoonGAN

Content loss introduced to ensure the resulting images retain semantic content of the input.

CartoonGAN uses a high-level feature map from a VGG network that pre-trained on ImageNet. It can preserve the content of objects.

$$
\begin{array}{l}
\mathcal{L}_{\text {con }}(G, D)= \mathbb{E}_{p_{i} \sim S_{\text {data }}(p)}\left[|| V G G_{l}\left(G\left(p_{i}\right)\right)-V G G_{l}\left(p_{i}\right) \|_{1}\right]
\end{array}
$$
Where:
* $l$ refers to the feature maps of a specific VGG layer.
* $p_{i} \in S_{d a t a}(p)$ is a photo.
* $G(p_{i})$ is a fake cartoon image that took a photo as input.
* the paper used the feature maps in `conv4_4` layer from a VGG network. 

## Total Objective function of CartoonGAN
$$
\mathcal{L}(G, D)=\mathcal{L}_{a d v}(G, D)+\omega \mathcal{L}_{\text {con }}(G, D)
$$
where the paper set $\omega = 10$.

# Initialization Phase of CartoonGAN

CartoonGAN proposed a new initialization phase to improve the convergence of the network.

> GAN model is highly nonlinear, with random
initialization, the optimization can be easily trapped at suboptimal local minimum.

The new initialization phase is done by:

* Pre-train the generator network with only content loss $\mathcal{L}_{\text {con }}$ for $N$ epochs, letting the generator to only reconstructs the content of input images.

The paper found this initialization phase helps CartoonGAN fast converge to a good configuration, without premature convergence.

# Architecture of CartoonGAN

## Architecture of Generator and Discriminator in CartoonGAN

> Refer to figure 2 of the Paper.

Key point:
* Generator: 8 residual blocks Encoder-Decoder like network
  * Batch Norm, ReLU
* Discriminator: PatchGAN
  * Batch Norm, LeakyReLU

### Generator Implementation

```python
import torch
import torch.nn as nn

# PyTorch implementation by vinesmsuic
# The paper claimed to use BatchNorm and Leaky ReLu.
# But here we use InstanceNorm instead of BatchNorm.

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        #Elementwise Sum (ES)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=8):
        super().__init__()
        self.initial = nn.Sequential(
            #k7n64s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="zeros"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        #Down-convolution
        self.down_blocks = nn.ModuleList(
            [
                #k3n128s2
                nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode="zeros"),
                #k3n128s1
                nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features*2),
                nn.ReLU(inplace=True),

                #k3n256s2
                nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1, padding_mode="zeros"),
                #k3n256s1
                nn.Conv2d(num_features*4, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features*4),
                nn.ReLU(inplace=True),
            ]
        )

        #8 residual blocks => 8 times [k3n256s1, k3n256s1]
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, padding=1) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up_blocks = nn.ModuleList(
            [
                #k3n128s1/2
                nn.ConvTranspose2d(num_features*4, num_features*2, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode="zeros"),
                #k3n128s1
                nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features*2),
                nn.ReLU(inplace=True),

                #k3n64s1/2
                nn.ConvTranspose2d(num_features*2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1, padding_mode="zeros"),
                #k3n64s1
                nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode="zeros"),
                nn.InstanceNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        )

        #Convert to RGB
        #k7n3s1
        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="zeros")

    def forward(self, x):
        x = self.initial(x)
        #Down-convolution
        for layer in self.down_blocks:
            x = layer(x)
        #8 residual blocks
        x = self.res_blocks(x)
        #Up-convolution
        for layer in self.up_blocks:
            x = layer(x)
        #Convert to RGB
        x = self.last(x)
        #TanH
        return torch.tanh(x)
        
```

### Discriminator Implementation

```python
import torch
import torch.nn as nn

# PyTorch implementation by vinesmsuic
# The paper claimed to use BatchNorm and Leaky ReLu.
# But here we use InstanceNorm instead of BatchNorm.
# We also use reflect padding instead of constant padding here (to reduce artifacts?)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.model = nn.Sequential(
            #k3n32s1
            nn.Conv2d(in_channels,features[0],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n64s2
            nn.Conv2d(features[0],features[1],kernel_size=3,stride=2,padding=1,padding_mode="reflect"), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s1
            nn.Conv2d(features[1],features[2],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.InstanceNorm2d(features[2]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s2
            nn.Conv2d(features[2],features[2],kernel_size=3,stride=2,padding=1,padding_mode="reflect"), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n256s1
            nn.Conv2d(features[2],features[3],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n256s1
            nn.Conv2d(features[3],features[3],kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n1s1
            nn.Conv2d(features[3],out_channels,kernel_size=3,stride=1,padding=1,padding_mode="reflect"), 
        )

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)

```


# Limitations of CartoonGAN
* Contains a lot of obvious "artifacts"
* Generate low resolution outputs
* Checkerboard effects
* Many people claimed could not reproduce the results?
