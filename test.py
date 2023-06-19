from generator_model import Generator
import config
import utils 
from mydataset import DustyDataset

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
from guidedfilter import guided_filter
import glob
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    
    g_s, g_m = utils.get_generator()
    opt_gs, opt_gm = utils.get_optim_generator(g_s, g_m)

    g_m, g_s, opt_gm, opt_gs = utils.loadCHECKPOINT_GENERATOR(g_m, g_s, opt_gm, opt_gs)

    test_dataset = DustyDataset(root_dir=config.TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for index , (test_image, img_file) in enumerate(test_loader):
        # print(img_file)
        test_image = test_image.to(config.DEVICE)
        out_m = g_m(test_image)
        out_s = g_s(test_image)
        # concat = torch.cat((test_image*0.5 + 0.5, dedusted*0.5 + 0.5), 2)
        save_image(out_m*.5+.5, f"results/final/master/{img_file}.png")
        save_image(out_s*0.5 + 0.5, f"results/final/supporter/{img_file}.png")
        print(f"saving both image and image-concatinate {img_file} has done successful")
        

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)
        
def gudedfilter():
    files = glob.glob("results/final/master/*.*")
    i=0
    for x in files:
        img = cv2.imread(x)
        filename = Path(x).stem
        img = gammaCorrection(img, 1.9)
        # img = cv2.detailEnhance(img, sigma_s=1, sigma_r=0.9)# gammaCorrection(img, 2.0)
        cv2.imwrite(f"results/final/gama_master/{filename}.jpg", img)
        i+=1
        print(x)
    
    # cv2.imshow('Original image', img)
    # cv2.imshow('Gamma corrected image', gammaImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def plot_histogram(hist):
    plt.plot(hist, color='blue')
    plt.xlim([0, 255])
    plt.ylim([0,10000000])
    plt.title('blue histogram')
    plt.show()
    
def histogram_images():
    path_dust = "results/test/gama_master/29.jpg"
    path_meode = "results/test/gama_master/29.jpg"
    
    img_model = cv2.imread(path_meode)
    
    red = cv2.calcHist([img_model], [0], None, [256], [0, 255])
    green = cv2.calcHist([img_model], [1], None, [256], [0, 255])
    blue = cv2.calcHist([img_model], [2], None, [256], [0, 255])
    plot_histogram(red)
    
    
def Traditional_resized():
    test_dataset = DustyDataset(root_dir="data/traditional methods/Mars")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for index , (test_image, img_file) in enumerate(test_loader):
        # print(img_file)
        test_image = test_image.to(config.DEVICE)
        save_image(test_image*.5+.5, f"data/Traditional resized/Mars/{img_file}.png")
        print(f"saving both image and image-concatinate {img_file} has done successful")
if __name__ == "__main__":
    # main()
    # gudedfilter()
    # histogram_images()
    Traditional_resized()