from dataset.synthia import SYNTHIA
import ME
from config import get_config 
import torch 
from torch.utils.data import DataLoader

args_data = {
    "dirs" : 
          {"project" : "/mnt/c/Users/matte/iCloudDrive/Documents/Studies/IASD/3DPoint_Cloud/Project",
            "RGB" : "data/SYNTHIA-SEQS-01-SPRING/SYNTHIA-SEQS-01-SPRING/RGB/Stereo_Left/Omni_F",
           "label" :  "data/SYNTHIA-SEQS-01-SPRING/SYNTHIA-SEQS-01-SPRING/GT/Stereo_Left/Omni_F"
           },
    "n_sample" : 10

           }

if __name__ == '__main__':
    args = get_config()
    train_dataset = SYNTHIA(args_data)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=ME.utils.batch_sparse_collate)
    

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in train_dataloader:
        print(batch)
    
