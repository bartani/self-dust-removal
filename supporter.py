from generator_model import Generator
from discriminator_model import Discriminator
import config
import torch.optim as optim
import torch
import torchvision.models as models
from mydataset import dataset, DustyDataset
from torch.utils.data import DataLoader
import utils  
from tqdm import tqdm
import torch.nn as nn
from torchvision.utils import save_image


def load_supporter(gen, opt_gen, disc, opt_disc):
    utils.load_checkpoint(config.CHECKPOINT_GEN_SUPPORTER, gen, opt_gen, config.LEARNING_RATE,)
    utils.load_checkpoint(config.CHECKPOINT_DISC_SUPPORTER, disc, opt_disc, config.LEARNING_RATE,)
    return gen, opt_gen, disc, opt_disc

def train_discriminator(disc, y_fake, x, y, bce, scaler, opt):
    # x = dusty-image y = clean-image 
    # Train Discriminator
    with torch.cuda.amp.autocast():
        
        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
    
    opt.zero_grad()
    scaler.scale(D_loss).backward()
    scaler.step(opt)
    scaler.update()
    
    return disc, opt, scaler, D_loss


def save_results(loader, g_m, idx, n_epoch):
    for index , x in enumerate(loader):
        x = x.to(config.DEVICE)
        y = g_m(x)
        concat = torch.cat((x*0.5 + 0.5, y*0.5 + 0.5), 2)
        save_image(concat, f"results/out_supporter2/{n_epoch}_{idx}_clean_{index}.png")
   
def save_model(gen, opt_gen, disc, opt_disc):
    utils.save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN_MASTER)
    
    utils.save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC_MASTER)

def train(gen, disc, opt_gen, opt_disc, scaler_gen, scaler_disc, dl_train, dl_valid, bce, l1_loss, n_epoch):
    
    loop = tqdm(dl_train, leave=True)
    for idx, (dust_img, clean_img, synthetic_dusty, dedusted_img) in enumerate(loop):
        
        dust_img = dust_img.to(config.DEVICE)
        clean_img = clean_img.to(config.DEVICE)
        synthetic_dusty = synthetic_dusty.to(config.DEVICE)
        dedusted_img = dedusted_img.to(config.DEVICE)
        
        y_fake = gen(dust_img)
        disc, opt_disc, scaler_disc, loss_disc = train_discriminator(disc, y_fake, dust_img, clean_img, bce, scaler_disc, opt_disc)

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(dust_img, y_fake.detach())
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))                        
            La = G_fake_loss
            L1 = l1_loss(y_fake, dedusted_img)  * 100          
            G_loss = La + L1

        opt_gen.zero_grad()
        scaler_gen.scale(G_loss).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        
        if idx % 1000 == 0:
            save_results(dl_valid, gen, idx, n_epoch)
        
        if idx % 10 == 0:
            loop.set_postfix(
                loss_disc = loss_disc.mean().item(),
                G_loss = G_loss.mean().item(),
            )
        
        

def main():
    
    # create discriminator and generator
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    
    # optimaizer
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    
    # scaler
    scaler_disc = torch.cuda.amp.GradScaler() 
    scaler_gen = torch.cuda.amp.GradScaler()
    
    # g_m, g_s, opt_gm, opt_gs = utils.loadCHECKPOINT_GENERATOR(g_m, g_s, opt_gm, opt_gs)
    # d_m, d_s, critic, opt_dm, opt_ds, opt_critic = utils.loadCHECKPOINT_DISRIMINATOR(d_m, d_s, critic, opt_dm, opt_ds, opt_critic)
    
    # create dataset
    dl_train = utils.get_dataloader()
    dl_valid = utils.get_validloader()
    
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    gen, opt_gen, disc, opt_disc = load_supporter(gen, opt_gen, disc, opt_disc)
    
    for epoch in range(2,4):
        train(gen, disc, opt_gen, opt_disc, scaler_gen, scaler_disc, dl_train, dl_valid, BCE, L1_LOSS, epoch)
        save_model(gen, opt_gen, disc, opt_disc)
        

def test():
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    utils.load_checkpoint(config.CHECKPOINT_GEN_SUPPORTER, gen, opt_gen, config.LEARNING_RATE,)
    dl = utils.get_testloader()
    
    for index , x in enumerate(dl):
        x = x.to(config.DEVICE)
        y = gen(x)
        concat = torch.cat((x*0.5 + 0.5, y*0.5 + 0.5), 2)
        save_image(concat, f"results/test_supporter/{index}.png")
        print(f"saving both image and image-concatinate {index} has done successful")
        
        
   
    
if __name__ == "__main__":
    main()
    # test()