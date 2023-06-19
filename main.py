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

def train_discriminator(disc, opt, scaler, base_img, fake, real, bce):
    # x = dusty-image y = clean-image 
    # Train Discriminator
    with torch.cuda.amp.autocast():
        
        D_real = disc(base_img, real)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(base_img, fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
    
    opt.zero_grad()
    scaler.scale(D_loss).backward()
    scaler.step(opt)
    scaler.update()
    
    return disc, opt, scaler, D_loss

def train_generator_supporter(gen, disc, opt, scaler, base_img, out_gen, clean_img, bce, l1_loss):
    # Train generator supporter
    with torch.cuda.amp.autocast():
        D_fake = disc(base_img, out_gen.detach())
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))                        
        La = G_fake_loss
        L1 = l1_loss(out_gen, clean_img) * 100         
        G_loss = La + L1
        
    opt.zero_grad()
    scaler.scale(G_loss).backward()
    scaler.step(opt)
    scaler.update()
    return gen, opt, scaler, G_loss


def train_generator_master(gen, disc, critic, opt, scaler, base_img_disc, base_img_critic, clean_img, out_gm, bce, l1_loss, pre_model):
    # Train generator supporter
    # x = synthetic-dusty-image y = clean-image 
    with torch.cuda.amp.autocast():
        D_fake = disc(base_img_disc, out_gm.detach())
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))

        C_fake = critic(base_img_critic, out_gm.detach())
        C_fake_loss = bce(C_fake, torch.ones_like(D_fake))
                        
        La = G_fake_loss * 10
        Lc = C_fake_loss * 50
        L_P = l1_loss(pre_model(out_gm), pre_model(clean_img)) * 1.0
        L1 = l1_loss(out_gm, clean_img) * 50           
        G_loss = La + L1 + Lc + L_P
        
    opt.zero_grad()
    scaler.scale(G_loss).backward()
    scaler.step(opt)
    scaler.update()
    return gen, opt, scaler, G_loss

def save_results(loader, g_m, g_s, idx, n_epoch):
    for index , x in enumerate(loader):
        x = x.to(config.DEVICE)
        y_m = g_m(x)
        y_s = g_s(x)
        concat = torch.cat((x*0.5 + 0.5, y_m*0.5 + 0.5, y_s*0.5 + 0.5), 3)
        save_image(concat, f"results/new/{n_epoch}_{idx}_clean_{index}.png")
   
def save_model(g_m, g_s, opt_gm, opt_gs, d_m, d_s, critic, opt_dm, opt_ds, opt_critic):
    utils.save_checkpoint(g_m, opt_gm, filename=config.CHECKPOINT_GEN_MASTER)
    utils.save_checkpoint(g_s, opt_gs, filename=config.CHECKPOINT_GEN_SUPPORTER)
    
    utils.save_checkpoint(d_m, opt_dm, filename=config.CHECKPOINT_DISC_MASTER)
    utils.save_checkpoint(d_s, opt_ds, filename=config.CHECKPOINT_DISC_SUPPORTER)
    utils.save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

def train(pre_model, g_m, g_s, d_m, d_s, critic, dl_train, dl_valid, bce, L1,
          scaler_ds, scaler_dm, scaler_critic, scaler_gs, scaler_gm,
          opt_ds, opt_dm, opt_critic, opt_gs, opt_gm, n_epoch):
    
    loop = tqdm(dl_train, leave=True)
    for idx, (dust_img, clean_img, synthetic_dusty, dedusted_img) in enumerate(loop):
        
        dust_img = dust_img.to(config.DEVICE)
        clean_img = clean_img.to(config.DEVICE)
        synthetic_dusty = synthetic_dusty.to(config.DEVICE)
        dedusted_img = dedusted_img.to(config.DEVICE)
        
        # out_gs = g_s(dust_img)
        # d_s, opt_ds, scaler_ds, loss_ds = train_discriminator(d_s, opt_ds, scaler_ds, dust_img, out_gs, clean_img, bce)
        # g_s, opt_gs, scaler_gs, loss_gs = train_generator_supporter(g_s, d_s, opt_gs, scaler_gs, dust_img, out_gs, dedusted_img, bce, L1)
        
        out_gm = g_m(synthetic_dusty)
        d_m, opt_dm, scaler_dm, loss_dm = train_discriminator(d_m, opt_dm, scaler_dm, synthetic_dusty, out_gm, clean_img , bce)
        
        critic, opt_critic, scaler_critic, loss_critic = train_discriminator(critic, opt_critic, scaler_critic, dust_img, g_m(dust_img), g_s(dust_img).detach(), bce)
        
        
        g_m, opt_gm, scaler_gm, loss_gm = train_generator_master(g_m, d_m, critic, opt_gm, scaler_gm, synthetic_dusty, dust_img, clean_img, out_gm, bce, L1, pre_model)
        
        if idx % 500 == 0:
            save_results(dl_valid, g_m, g_s, idx, n_epoch)
        if idx % 1000 == 0:
            save_model(g_m, g_s, opt_gm, opt_gs, d_m, d_s, critic, opt_dm, opt_ds, opt_critic)
        
        if idx % 10 == 0:
            loop.set_postfix(
                # loss_ds = loss_ds.mean().item(),
                # loss_gs = loss_gs.mean().item(),
                loss_dm = loss_dm.mean().item(),
                loss_critic = loss_critic.mean().item(),
                loss_gm = loss_gm.mean().item(),
            )

        
        

def main():
    # create discriminator and generator
    d_s, d_m, critic = utils.get_discriminator()
    g_s, g_m = utils.get_generator()
    
    pre_model = models.resnet34(weights='ResNet34_Weights.DEFAULT').to(config.DEVICE)
    
    # optimaizer
    opt_ds, opt_dm, opt_critic = utils.get_optim_discriminator(d_s, d_m, critic)
    opt_gs, opt_gm = utils.get_optim_generator(g_s, g_m)
    
    # g_s, opt_gs, d_s, opt_ds = load_supporter(g_s, opt_gs, d_m, opt_dm)
    
    # scaler
    scaler_ds, scaler_dm, scaler_critic = utils.get_scaler_discriminator()    
    scaler_gs, scaler_gm = utils.get_scaler_generator()
    
    g_m, g_s, opt_gm, opt_gs = utils.loadCHECKPOINT_GENERATOR(g_m, g_s, opt_gm, opt_gs)
    d_m, d_s, critic, opt_dm, opt_ds, opt_critic = utils.loadCHECKPOINT_DISRIMINATOR(d_m, d_s, critic, opt_dm, opt_ds, opt_critic)
    
    # create dataset
    dl_train = utils.get_dataloader()
    dl_valid = utils.get_validloader()
    
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    for epoch in range(3,5):
        train(pre_model, g_m, g_s, d_m, d_s, critic, dl_train, dl_valid, BCE, L1_LOSS,
              scaler_ds, scaler_dm, scaler_critic, scaler_gs, scaler_gm,
              opt_ds, opt_dm, opt_critic, opt_gs, opt_gm, epoch)
        # save_model(g_m, g_s, opt_gm, opt_gs, d_m, d_s, critic, opt_dm, opt_ds, opt_critic)
        


   
    
if __name__ == "__main__":
    main()