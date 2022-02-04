# PyTorch-CartoonGAN

Unofficial PyTorch implementation of CartoonGAN. We followed the original Lua training implementation from the paper author ([Yang Chen](https://github.com/FlyingGoblin/CartoonGAN)).

# Repo Structure
```python
├─checkpoints
├─data
│  ├─train 
│  │  ├─cartoon # You put cartoon images here
│  │  ├─cartoon_edge_pair 
│  │  └─photo   # You put photo images here
│  └─val
│      └─photo # You put photo images here
└─results
    ├─test  
    └─train
```


# Dependenies
* Albumentations
```bash
pip install -U albumentations
```
* tqdm
```bash
pip install tqdm
```

# To start training

0. Read `REVIEW.md` to understand what it is
1. Prepare the photo and cartoon data
2. Get the pre-trained VGG19 weight and put it in the `CartoonGAN` folder : 
    https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
3. Preprocess data through edge promoting
```bash
python edge_smooth.py
```
4. Edit `config.py`
* Set T
* If you are using your custom data that are in random size, please enable RandomCrop in `config.py`.
```python
transform_cartoon_pairs = A.Compose(
    #additional_targets: apply same augmentation on both images
    [   
        A.RandomCrop(width=IMAGE_SIZE*1.2, height=IMAGE_SIZE*1.2),
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ], 
    additional_targets={"image0": "image"},
)
```
5. Training
```bash
python train.py
```
* The training consist of initialization phase and training phase.
6. Wait for a long time and see the results at `results` folder


# TODO

[] LR Scheduler
[] Loss visualization
[] WandB visualization
[] Inference Code
[X] Explaining Code

# Working Environments

* Windows with CUDA
* Ubuntu with CUDA

