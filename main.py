from dataset.synthia import SYNTHIA
import ME
from config import get_config 
import torch 
from torch.utils.data import DataLoader


if __name__ == '__main__':
    args = get_config()
    train_dataset = SYNTHIA(args)
    indoor_seg_train = Indoor3DSemSeg(data_dir=cfg['data']['path'],
                                  train=True,
                                  num_points=cfg['data']['num_points'],
                                  aug=aug,
                                  test_area=cfg['data']['test_area'],
                                  data_precent=cfg['data']['data_percent'])

indoor_seg_val = Indoor3DSemSeg(data_dir=cfg['data']['path'],
                                train=False,
                                num_points=cfg['data']['num_points'],
                                test_area=cfg['data']['test_area'],
                                data_precent=cfg['data']['data_percent'])

train_sampler = torch.utils.data.distributed.DistributedSampler(indoor_seg_train)
val_sampler = torch.utils.data.distributed.DistributedSampler(indoor_seg_val)

dataloader_train = torch.utils.data.DataLoader(indoor_seg_train,
                                               batch_size=cfg['data']['batch_size'],
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg['data']['num_workers'],
                                               pin_memory=False, sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

dataloader_val = torch.utils.data.DataLoader(indoor_seg_val,
                                             batch_size=cfg['data']['batch_size_val'],
                                             shuffle=False,
                                             num_workers=cfg['data']['num_workers'],
                                             pin_memory=False, sampler=val_sampler,
                                             worker_init_fn=worker_init_fn)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=ME.utils.batch_sparse_collate)
    

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    for batch in train_dataloader:
        print(batch)
    
