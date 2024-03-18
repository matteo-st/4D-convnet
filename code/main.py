from dataset.synthia import SYNTHIA
import ME
from config import get_config 
import torch 
from torch.utils.data import DataLoader


if __name__ == '__main__':
    args = get_config()
    train_dataset = SYNTHIA(args)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=ME.utils.batch_sparse_collate)
    

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in train_dataloader:
        print(batch)
    
