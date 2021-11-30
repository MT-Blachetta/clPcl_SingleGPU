"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
import copy
#from spherecluster import VonMisesFisherMixture
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from utils.utils import AverageMeter, ProgressMeter


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)  # == stl-10.example: € Tensor[2*batch_size, 3, 96, 96] c=3
        input_ = input_.cuda(non_blocking=True)

        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1) # == € Tensor[2*batch_size, 128].view(b,2,-1)
        loss = criterion(output)

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)
            
def pcl_cld_train(train_loader, instance_branch, group_branch, criterion, optimizer, epoch, M_num_clusters):
    
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[losses],prefix="Epoch: [{}]".format(epoch))
    alpha = 0.05
    iloss = torch.nn.CrossEntropyLoss()
    iloss = iloss.cuda()
        
    instance_branch.train()
    group_branch.train()
    #originImage_batch = batch['image']
#augmentedImage_batch = batch['image_augmented']
#originImage_batch = originImage_batch.cuda(non_blocking=True)
#augmentedImage_batch = augmentedImage_batch.cuda(non_blocking=True)
#print("batch_image_shape: "+str(originImage_batch.shape))

    for i, batch in enumerate(train_loader):
        originImage_batch = batch['image']
        augmentedImage_batch = batch['image_augmented']
        
        originImage_batch = originImage_batch.cuda(non_blocking=True)
        augmentedImage_batch = augmentedImage_batch.cuda(non_blocking=True)

        logits, labels = instance_branch(originImage_batch,augmentedImage_batch)
        instance_loss = iloss(logits,labels)

#M_num_clusters = [2,4,8,16]
        original_view = group_branch(originImage_batch)
        augmented_view = group_branch(augmentedImage_batch)
        feature_dim = len(original_view[0])
        batch_size = len(original_view)
        M_kmeans_results = []
        MI_kmeans_results = []
        concentration_matrices = []
        concentration_matrices_I = []
        M_labels = []
        M_labels_I = []

        alpha = 0.1
        divzero = 0.1
        ov = original_view.cpu().detach().numpy()
        print(ov.shape)
        av = augmented_view.cpu().detach().numpy()

        for k in M_num_clusters:
            #from spherecluster import SphericalKMeans
            clusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=20, normalise=True, avoid_empty_clusters=True)
            #skm = SphericalKMeans(n_clusters=k)
            #skm.fit(ov)
            clusterer_I = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=20, normalise=True, avoid_empty_clusters=True)
            labels_ = clusterer.cluster(ov,True)
            labels_I = clusterer_I.cluster(av,True)
            cluster_centers = torch.Tensor( np.array(clusterer.means()) ) 
            cluster_centers_I = torch.Tensor( np.array(clusterer_I.means()) )
             
            #skm_I = SphericalKMeans(n_clusters=k)
            #skm_I.fit(av)

            M_kmeans_results.append( cluster_centers )
            MI_kmeans_results.append( cluster_centers_I )
            # c -> k
            center = [ cluster_centers[i] for i in range(k) ]
            center_I = [ cluster_centers_I[i] for i in range(k) ]
            cdat = [ x.unsqueeze(0).expand(batch_size,feature_dim) for x in center]
            cmatrix = torch.cat(cdat,1)
            cdat_I = [ x.unsqueeze(0).expand(batch_size,feature_dim) for x in center_I]
            cmatrix_I = torch.cat(cdat_I,1)

            original_cpu = original_view.cpu()
            augmented_cpu = augmented_view.cpu()          
            fmatrix = torch.Tensor(copy.deepcopy(ov))
            fmatrix_I = torch.Tensor(copy.deepcopy(av))
  #fmatrix = copy.deepcopy(original_cpu)
  #fmatrix_I = copy.deepcopy(augmented_cpu)

            for _ in range(1,k): fmatrix = torch.cat((fmatrix,original_cpu),1)
            for _ in range(1,k): fmatrix_I = torch.cat((fmatrix_I,augmented_cpu),1)
                
            cmatrix = cmatrix.cuda()
            fmatrix = fmatrix.cuda()
            cmatrix_I = cmatrix_I.cuda()
            fmatrix_I = fmatrix_I.cuda()
            
            zmatrix = fmatrix-cmatrix
            zmatrix = zmatrix*zmatrix
            result = zmatrix.flatten(0).view(batch_size,k,feature_dim)
            result = torch.sum(result,2)
            result = torch.sqrt(result)

            zmatrix_I = fmatrix_I-cmatrix_I
            zmatrix_I = zmatrix_I*zmatrix_I
            result_I = zmatrix_I.flatten(0).view(batch_size,k,feature_dim)
            result_I = torch.sum(result_I,2)
            result_I = torch.sqrt(result_I)
            
            assign = torch.zeros(batch_size,k)
            assign_I = torch.zeros(batch_size,k)

            for i in range(batch_size):
              assign[i][ int(skm.labels_[i]) ] = 1
              assign_I[i][ int(skm_I.labels_[i]) ] = 1
                
            assign = assign.cuda()
            assign_I = assign_I.cuda()
            
            avgDistance = torch.sum(assign*result,0)
            Z = torch.sum(assign,0) + 1
            Zlog = torch.log(Z+alpha)
            divisor = Z*Zlog
            concentrations = (avgDistance/divisor) + divzero
            concentrations = concentrations.cpu()
            #avgDistance = avgDistance.cuda()
            #divisor = divisor.cuda()
            avgDistance_I = torch.sum(assign_I*result_I,0)
            Z_I = torch.sum(assign_I,0) + 1
            Zlog_I = torch.log(Z_I+alpha)
            divisor_I = Z_I*Zlog_I
            concentrations_I = (avgDistance_I/divisor_I) + divzero
            concentrations_I = concentrations_I.cpu()
            
            concentration_matrices.append(concentrations)
            concentration_matrices_I.append(concentrations_I)
            
            M_labels.append( labels_ )
            M_labels_I.append( labels_I )
#-------------------------------------------------------------------------------------------------------
#group_loss = pcl_cld_loss(original_view,augmented_view,M_kmeans_results,MI_kmeans_results,concentration_matrices,concentration_matrices_I)
        group_loss = criterion( features = original_view, features_I = augmented_view, M_kmeans = M_kmeans_results , M_kmeans_I = MI_kmeans_results, concentrations = concentration_matrices, concentrations_I = concentration_matrices_I,labels = M_labels, labels_I = M_labels_I, lb = 1)
        
        loss = instance_loss + group_loss
        
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
