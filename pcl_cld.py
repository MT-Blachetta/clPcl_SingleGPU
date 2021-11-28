import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_backbone_model ,get_instance_model,get_group_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate, get_clustering
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import pcl_cld_train
from utils.utils import fill_memory_bank
from termcolor import colored

""" <<INPUT>> """
# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

"""-----------"""

def main():

    #1# Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    ###
    
    
    #2# Model - retrieve the model from config file                        OK
    print(colored('Retrieve model', 'blue'))
    
    backbone = get_backbone_model(p)
    print('Model is {}'.format(backbone.__class__.__name__))
    #print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in backbone.parameters()) / 1e6))
    print(backbone)
    
    
    instance_model = get_instance_model(p, backbone)
    instance_head = instance_model.encoder_q.contrastive_head
    
    group_model = get_group_model(p, backbone)
    group_head = group_model.contrastive_head
    print('Model is {}'.format(instance_model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in instance_model.parameters()) / 1e6))
    print(instance_model)
    print('Model is {}'.format(group_model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in group_model.parameters()) / 1e6))
    print(group_model)
    instance_model = instance_model.cuda()
    group_model = group_model.cuda()
   
     #> CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True    
    ###
    
    
    #3# Dataset                                                       OK
    #A - get transformormations for the dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p) 
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    
    #B - get Dataset from files
    print('Validation transforms:', val_transforms)
    split_ = 'train'
    if p['train_db_name'] == 'stl-10':
        split_ = 'train+unlabeled'
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                        split=split_) # Split is for stl-10
                                        
    val_dataset = get_val_dataset(p, val_transforms)
    
    #C - put the dataset to the dataloader for training purposes
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
    ###


    #4# Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train') # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset) 
    
    memory_bank_base = MemoryBank(len(base_dataset), 
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()
    ###


    #5# Training Parameter                                              OK
    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler                                           OK
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, group_model)
    print(optimizer)
    ###
    
    M_num_clusters = get_clustering(p)
 
    #6# Checkpoint to continue last training phase                             OK
    if os.path.exists(p['pretext_checkpoint_backbone']):
        print(colored('Restart from checkpoint (backbone) {}'.format(p['pretext_checkpoint_backbone']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint_backbone'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        backbone.load_state_dict(checkpoint['model'])
        #backbone.cuda()
        start_epoch = checkpoint['epoch']
    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
      
  
    if os.path.exists(p['pretext_checkpoint_instance']):
        print(colored('Restart from checkpoint (instance_model) {}'.format(p['pretext_checkpoint_instance']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint_instance'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        instance_head # instance_model.encoder_q.contrastive_head
        instance_model.load_state_dict(checkpoint['model'])
        instance_model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        instance_model = instance_model.cuda()
        
    if os.path.exists(p['pretext_checkpoint_group']):
        print(colored('Restart from checkpoint (group_model) {}'.format(p['pretext_checkpoint_group']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint_group'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        group_model.load_state_dict(checkpoint['model'])
        group_model.cuda()
        start_epoch = checkpoint['epoch']
        
    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        group_model = group_model.cuda()
        
    ###
    
    #7# Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        #a - Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        #b - Train the model with the clPcl method for one epoch (iteration)
        print('Train ...')
        pcl_cld_train(train_loader = train_dataloader, instance_branch = instance_model, group_branch = group_model, criterion = criterion, optimizer = optimizer, epoch = epoch, M_num_clusters = M_num_clusters)

        #c - Fill memory bank (Data Structure for nearest neighbors of input-instances)
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, group_model, memory_bank_base)

        #d - Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, group_model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' %(top1)) 
        
        #e - Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': instance_model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint_instance'])
                    
        torch.save({'optimizer': optimizer.state_dict(), 'model': group_model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint_group'])
                    
        

    # Save final model
    torch.save(instance_model.state_dict(), p['pretext_model_instance'])
    torch.save(group_model.state_dict(),p['pretext_model_group'])
    ###
    """ <doc>
    torch.save(obj, f: Union[str, os.PathLike, BinaryIO], 
                pickle_module=<module 'pickle' from '/opt/conda/lib/python3.6/pickle.py'>, 
                pickle_protocol=2, _use_new_zipfile_serialization=True) → None
                
    Saves an object to a disk file
    
    Parameters:

        obj – saved object

        f – a file-like object (has to implement write and flush) or a string or os.PathLike object containing a file name

        pickle_module – module used for pickling metadata and objects

        pickle_protocol – can be specified to override the default protocol


    </doc> """
    
    
    
    
    #8# determine nearest neighbors
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(base_dataloader, group_model, memory_bank_base)
    topk = 20
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_train_path'], indices)   

   
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_dataloader, group_model, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)   
    ###
 
if __name__ == '__main__':
    main()
