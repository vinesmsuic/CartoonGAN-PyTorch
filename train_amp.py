import torch
import torch.nn as nn
import torch.optim as optim
import config
import os
from dataset import TrainDataset, TestDataset
from generator_model import Generator
from discriminator_model import Discriminator
from VGGNet import VGGNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from utils import save_test_examples, load_checkpoint, save_checkpoint

# This code throw error because it cause NaN values... how do i solve it?

def initialization_phase(gen, loader, opt_gen, l1_loss, g_scaler, VGG, pretrain_epochs):
    for epoch in range(pretrain_epochs):
        loop = tqdm(loader, leave=True)
        
        for idx, (sample_photo, _, _) in enumerate(loop):
            sample_photo = sample_photo.to(config.DEVICE)

            # train generator G
            with torch.cuda.amp.autocast():
                reconstructed = gen(sample_photo)

                sample_photo_feature = VGG(sample_photo)
                reconstructed_feature = VGG(reconstructed)
                reconstruction_loss = config.LAMBDA_CONTENT * l1_loss(reconstructed_feature, sample_photo_feature.detach())

            opt_gen.zero_grad()
            g_scaler.scale(reconstruction_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()    

            loop.set_postfix(epoch=epoch)
        
    save_image(sample_photo*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, "0_initialization_phase_photo.png"))
    save_image(reconstructed*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, "0_initialization_phase_reconstructed.png"))
    


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, mse, g_scaler, d_scaler, VGG):
    loop = tqdm(loader, leave=True)

    # Training
    for idx, (sample_photo, sample_cartoon, sample_edge) in enumerate(loop):
        sample_photo = sample_photo.to(config.DEVICE)
        sample_cartoon = sample_cartoon.to(config.DEVICE)
        sample_edge = sample_edge.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():

            #Pass samples into Discriminator: Fake Cartoon, real Cartoon and Edge
            fake_cartoon = gen(sample_photo)

            D_real = disc(sample_cartoon)
            D_fake = disc(fake_cartoon.detach())
            D_edge = disc(sample_edge)

            #Compute loss (edge-promoting adversarial loss)
            D_real_loss = mse(D_real, torch.ones_like(D_real))
            D_fake_loss = mse(D_fake, torch.zeros_like(D_fake))
            D_edge_loss = mse(D_edge, torch.zeros_like(D_edge))

            # Author's code divided it by 3.0, I believe it has similar thoughts to CycleGAN (divided by 2 with only 2 loss)
            D_loss = D_real_loss + D_fake_loss + D_edge_loss / 3.0   
            
        opt_disc.zero_grad() # clears old gradients from the last step
        d_scaler.scale(D_loss).backward() #backpropagation
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_real = disc(sample_cartoon)
            D_fake = disc(fake_cartoon.detach())

            G_fake_loss = mse(D_fake, torch.ones_like(D_fake))

            # Content loss
            sample_photo_feature = VGG(sample_photo)
            fake_cartoon_feature = VGG(fake_cartoon)
            content_loss = l1_loss(fake_cartoon_feature, sample_photo_feature.detach())

            # Compute loss (adversarial loss + lambda*content loss)
            G_loss = G_fake_loss + config.LAMBDA_CONTENT * content_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(sample_photo*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, "step_" + str(idx) + "_photo.png"))
            save_image(fake_cartoon*0.5+0.5, os.path.join(config.RESULT_TRAIN_DIR, "step_" + str(idx) + "_fakecartoon.png"))

        #loop.set_postfix(step=idx)


def main():
    print(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    VGG19 = VGGNet(in_channels=3, VGGtype="VGG19", init_weights=config.VGG_WEIGHTS, batch_norm=False, feature_mode=True)
    VGG19 = VGG19.to(config.DEVICE)
    VGG19.eval()

    if config.LOAD_MODEL:
        is_gen_loaded = load_checkpoint(
            gen, opt_gen, config.LEARNING_RATE, folder=config.CHECKPOINT_FOLDER, checkpoint_file=config.LOAD_CHECKPOINT_GEN
        )
        is_disc_loaded = load_checkpoint(
            disc, opt_disc, config.LEARNING_RATE, folder=config.CHECKPOINT_FOLDER, checkpoint_file=config.LOAD_CHECKPOINT_DISC
        )
    
    #BCE_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()
    MSE_Loss = nn.MSELoss() # went through the author's code and found him using LSGAN, LSGAN should gives better training

    
    train_dataset = TrainDataset(config.TRAIN_PHOTO_DIR, config.TRAIN_CARTOON_EDGE_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = TestDataset(config.VAL_PHOTO_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)

    #Use float64 training 
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Initialization Phase
    if not(is_gen_loaded and is_disc_loaded):
        print("="*80)
        print("=> Initialization Phase")
        initialization_phase(gen, train_loader, opt_gen, L1_Loss, g_scaler, VGG19, pretrain_epochs=config.PRETRAIN_EPOCHS)
        print("Finished Initialization Phase")
        print("="*80)

    # Do the training
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_Loss, MSE_Loss, g_scaler, d_scaler, VGG19)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, epoch, folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, epoch, folder=config.CHECKPOINT_FOLDER, filename=config.CHECKPOINT_DISC)

        # Test Some data
        save_test_examples(gen, val_loader, epoch, folder=config.RESULT_TEST_DIR)

if __name__ == "__main__":
    main()