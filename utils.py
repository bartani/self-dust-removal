import torch
import config
from torchvision.utils import save_image
import numpy as np
import torchvision.transforms as T
from PIL import Image
from generator_model import Generator
from discriminator_model import Discriminator
import torch.optim as optim
from mydataset import dataset, DustyDataset
from torch.utils.data import DataLoader

def get_generator():
    # create generator
    supporter = Generator(in_channels=3, features=64).to(config.DEVICE)
    master = Generator(in_channels=3, features=64).to(config.DEVICE)
    return supporter, master

def get_discriminator():
    # create discraminator
    supporter = Discriminator(in_channels=3).to(config.DEVICE)
    master = Discriminator(in_channels=3).to(config.DEVICE)
    critic = Discriminator(in_channels=3).to(config.DEVICE)  # center discraminator
    return supporter, master, critic

def get_optim_discriminator(supporter, master, critic):
    opt_supporter = optim.Adam(supporter.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_master = optim.Adam(master.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    return opt_supporter, opt_master, opt_critic

def get_optim_generator(supporter, master):
    opt_supporter = optim.Adam(supporter.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_master = optim.Adam(master.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    return opt_supporter, opt_master

def get_scaler_discriminator():
    supporter = torch.cuda.amp.GradScaler()
    master = torch.cuda.amp.GradScaler()
    critic = torch.cuda.amp.GradScaler()
    return supporter, master, critic

def get_scaler_generator():
    supporter = torch.cuda.amp.GradScaler()
    master = torch.cuda.amp.GradScaler()
    return supporter, master

def get_dataloader():
    myds = dataset(config.DUST_PATH, config.CLEAN_PATH, config.DEDUSTED_PATH)
    loader = DataLoader(
        myds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    return loader

def get_validloader():
    val_dataset = DustyDataset(root_dir=config.VALID_PATH)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return val_loader

def get_testloader():
    test_dataset = DustyDataset(root_dir=config.TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

def loadCHECKPOINT_GENERATOR(g_m, g_s, opt_gm, opt_gs):
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_MASTER, g_m, opt_gm, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_SUPPORTER, g_s, opt_gs, config.LEARNING_RATE,
        )
        
    return g_m, g_s, opt_gm, opt_gs

def loadCHECKPOINT_DISRIMINATOR(d_m, d_s, critic, opt_dm, opt_ds, opt_critic):
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_DISC_MASTER, d_m, opt_dm, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_SUPPORTER, d_s, opt_ds, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )
        
    return d_m, d_s, critic, opt_dm, opt_ds, opt_critic


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_dark_channel(I, w=15):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, [(w // 2, w // 2), (w // 2, w // 2), (0, 0)], 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch

def get_torh_to_image(x):
    transform = T.ToPILImage()
    img = transform(x)
    return img
def get_gen_dark_chanel(x):
    transform = T.ToPILImage()
    img = transform(x)
    print(type(img))
    return img
    

