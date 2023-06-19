from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
import random
import torch
from torchvision.utils import save_image
import math

from tqdm import tqdm
from pathlib import Path

def synthetic_dusty_image(org):
    J = np.asarray(org)/255.0

    I = np.zeros(J.shape)
    d = -1.0
    e = random.uniform(1.0,2.5)
    #T = math.exp(e*d)
    A = [random.uniform(.65,.85), random.uniform(.4,.55), random.uniform(.20,.35)]
    
    Tr = math.exp(1*e*d)
    Tg = math.exp(1*e*d) # need to change value for realstic dusty images creation
    Tb = math.exp(1*e*d)
    
      #Tr = math.exp(1.0*e*d)
      #Tg = math.exp(1.0*e*d)   #Red
      #Tb = math.exp(0.8*e*d)
    
    I[:,:,0] = (J[:,:,0]*Tr)+A[0]*(1-Tr)
    I[:,:,1] = (J[:,:,1]*Tg)+A[1]*(1-Tg)
    I[:,:,2] = (J[:,:,2]*Tb)+A[2]*(1-Tb)
    dust_img = Image.fromarray((I*255).astype(np.uint8)).convert('RGB')
    
    return dust_img

class dataset(Dataset):
    def __init__(self, root_dust, root_clean, root_dedusted):
        self.root_dust = root_dust
        self.root_clean = root_clean
        self.root_dedusted = root_dedusted

        self.dust_images = os.listdir(root_dust)
        self.clean_images = os.listdir(root_clean)
        
        self.length_dataset = max(len(self.dust_images), len(self.clean_images)) 
        self.dust_len = len(self.dust_images)
        self.clean_len = len(self.clean_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        idx_dust = random.randint(0, self.dust_len-1)
        # print(index, idx_dust)
        dust_img = self.dust_images[idx_dust]
        clean_img = self.clean_images[index % self.clean_len]

        dust_path = os.path.join(self.root_dust, dust_img)
        clean_path = os.path.join(self.root_clean, clean_img)
        dedusted_path = os.path.join(self.root_dedusted, dust_img)

        dust_img = Image.open(dust_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        dedusted_img = Image.open(dedusted_path).convert("RGB")
        
        synthetic_dusty = synthetic_dusty_image(clean_img)
        
        # transforms
        
        # clean_img, dust_img, synthetic_dusty, dedusted_img = config.get_tfsm(clean_img, dust_img, synthetic_dusty, dedusted_img)

        dust_img = config.transforms(dust_img)
        clean_img = config.transforms(clean_img)
        synthetic_dusty = config.transforms(synthetic_dusty)
        dedusted_img = config.transforms(dedusted_img)

        return dust_img, clean_img, synthetic_dusty, dedusted_img
    

class DustyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        filename = Path(img_path).stem
        
        dusty_ = Image.open(img_path).convert('RGB')
        
        # perform any required transformation
        dusty_ = config.transforms(dusty_)

        return dusty_, filename
    
    
if __name__ == "__main__":
    myds = dataset(config.DUST_PATH, config.CLEAN_PATH, config.DEDUSTED_PATH)
    dl = DataLoader(myds, batch_size=1)
    loop = tqdm(dl, leave=True)
    for idx, (dust_img, clean_img, synthetic_dusty, dedusted_img) in enumerate(loop):
        
        if idx==1:
            break
        # concat_cover = torch.cat((clean_img*.5+.5, synthetic_dusty*.5+.5, dust_img*.5+.5, dedusted_img*.5+.5), 2)
        # save_image(concat_cover, f"results/datasample/sample_{idx}.png")
        save_image(clean_img*.5+.5, f"results/datasample/sample_C_{idx}.png")
        save_image(synthetic_dusty*.5+.5, f"results/datasample/sample_S_{idx}.png")
        save_image(dust_img*.5+.5, f"results/datasample/sample_D{idx}.png")
        save_image(dedusted_img*.5+.5, f"results/datasample/sample_O{idx}.png")
        # loop.set_postfix(
        #     IDX = idx,
        # )




