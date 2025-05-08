from pathlib import Path
from torch.utils.data import DataLoader
from common.custom_types import AFADataset
from common.registry import AFA_DATASET_REGISTRY


def prepare_datasets(train_dataset, val_dataset, batch_size: int):
    d_in = train_dataset.features.shape[-1]
    d_out = train_dataset.labels.shape[-1]
    
    for ds in (train_dataset, val_dataset):
        ds.features = ds.features.float()
        ds.labels = ds.labels.argmax(dim=1).long()
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset,   batch_size=batch_size,
                              pin_memory=True)
    
    return train_loader, val_loader, d_in, d_out
