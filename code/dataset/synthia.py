import os
from torch.utils.data.dataset import Dataset
import cv2

config = {
    "dirs" : 
          {"project" : "/mnt/c/Users/matte/iCloudDrive/Documents/Studies/IASD/3DPoint_Cloud/Project",
            "RGB" : "data/SYNTHIA-SEQS-01-SPRING/SYNTHIA-SEQS-01-SPRING/RGB/Stereo_Left/Omni_F",
           "label" :  "data/SYNTHIA-SEQS-01-SPRING/SYNTHIA-SEQS-01-SPRING/GT/Stereo_Left/Omni_F"
           },
    "n_sample" : 10

           }


import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME

class SYNTHIA(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.get_files(cfg)
        self.quantization_size = cfg['quantization_size']

    def get_files(self, cfg):
        RGB_files_list = os.listdir(cfg["dirs"]["RGB"])
        label_files_list = os.listdir(cfg["dirs"]["label"])
        
        if cfg["n_sample"] is not None:
            RGB_files_list = RGB_files_list[:cfg["n_sample"]]
            label_files_list = label_files_list[:cfg["n_sample"]]
        
        self.img_files = RGB_files_list
        self.label_files = label_files_list
    
    def __getitem__(self, index):
        # Load the RGB image and its corresponding label
        rgb_img_path = os.path.join(self.cfg["dirs"]["RGB"], self.img_files[index])
        label_img_path = os.path.join(self.cfg["dirs"]["label"], self.label_files[index])
        
        rgb_img = cv2.imread(rgb_img_path)  # cv2 loads image in BGR
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)  # Load label image as grayscale
        print("Label shape", label_img.shape)
        print("rgb_img", rgb_img.shape)
        
        # Preprocess the image and label if necessary (e.g., resizing)

        # Generate coordinates for each pixel
        coords = np.array(np.where(rgb_img)).transpose()
        coords = coords[:, [1, 2, 0]]  # Rearrange coords to (x, y, channel)

        # Flatten the RGB channels to fit into the 'feats' array
        feats = rgb_img.reshape(-1, 3) / 255.0  # Normalize features to [0, 1]

        # Flatten the label image to create labels array
        labels = label_img.flatten()

        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coords=coords,
            feats=feats,
            labels=labels,
            quantization_size=self.quantization_size)

        return discrete_coords, unique_feats, unique_labels
    
    def __len__(self):
        return len(self.img_files)



if __name__ == '__main__':
     os.chdir(config["dirs"]["project"])
     
