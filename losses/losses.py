"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold    
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations 
        output: cross entropy 
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak) 
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold 
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None
        
        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean') 
        
        return loss


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss
    
class PclCldLoss(nn.Module):
    
    def __init__(self):
        super(PclCldLoss, self).__init__()
        
    def forward(self,features,features_I,M_kmeans,M_kmeans_I,concentrations,concentrations_I,labels,labels_I,lb):
        
        M_num = len(concentrations)
        print(M_num)
        batch_size = features.size()[0]

        M_logits = []
        M_logits_I = []

        #if k == 2: print()

        for k in range(M_num):
          c = len(concentrations[k]) # c = num_clusters of Mk
          M_cmatrix = torch.zeros(c,batch_size)
          MI_cmatrix = torch.zeros(c,batch_size)
          for i in range(c):
            M_cmatrix[i,:] = 1/concentrations[k][i]
            MI_cmatrix[i,:] = 1/concentrations_I[k][i]

          #if k == 2: print(M_cmatrix)          
          M_cmatrix = M_cmatrix.cuda()
          MI_cmatrix = MI_cmatrix.cuda()     
          centroids = M_kmeans[k].cuda()
          centroids_I = M_kmeans_I[k].cuda()
          gLoss_or = torch.mm(centroids,features_I.T) # OK 
          gLoss_au = torch.mm(centroids_I,features.T)
          #print("gLoss_or type: "+str(type(gLoss_or)) )
          #print("gLoss_or shape: "+str(gLoss_or.shape) )
        #--------------------------------------------------------
          summing_logits = gLoss_or * M_cmatrix # OK
          summing_logits_I = gLoss_au * MI_cmatrix

          exp_logits = torch.exp(summing_logits)
          exp_logits_I = torch.exp(summing_logits_I)
          log_sum = torch.sum(exp_logits,0)
          #print("log_sum type: "+str(type(log_sum)))
          #print("log_sum shape: "+str(log_sum.shape))
          log_sum_I = torch.sum(exp_logits_I,0)

          positive_pair = torch.zeros(batch_size)
          positive_pair_I = torch.zeros(batch_size)

          exlogCPU = exp_logits.cpu()
          exlogCPU_I = exp_logits_I.cpu()
          #lcpu = labels[k].cuda()
          #lcpu_ = labels_I[k].cuda()
          for l in range(batch_size):
            positive_pair[l] = exlogCPU[int(labels[k][l])][l]
            positive_pair_I[l] = exlogCPU_I[int(labels_I[k][l])][l]

          positive_pair = positive_pair.cuda()
          positive_pair_I = positive_pair_I.cuda()
                    #positive_pair = torch.exp(torch.mm(positive_pair,gLoss_or))
                    #positive_pair_I = torch.exp(torch.mm(positive_pair_I,gLoss_au))

          M_logits.append( torch.sum( torch.log(positive_pair/log_sum) ).cpu() ) # +0.0001 ),0).cpu()       ) 
          M_logits_I.append( torch.sum( torch.log(positive_pair_I/log_sum_I) ).cpu() ) # +0.0001 ),0).cpu() ) 

        return (1/batch_size)*lb*(-1/M_num)*0.5*( sum(M_logits) + sum(M_logits_I) )

