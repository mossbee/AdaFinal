import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss implementation for face verification.
    
    Args:
        margin (float): Margin for triplet loss
        distance_metric (str): Distance metric to use ('cosine' or 'euclidean')
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, margin=0.3, distance_metric='cosine', reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings [batch_size, embedding_dim]
            positive (torch.Tensor): Positive embeddings [batch_size, embedding_dim]
            negative (torch.Tensor): Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: Triplet loss value
        """
        if self.distance_metric == 'cosine':
            # Normalize embeddings for cosine distance
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            
            # Cosine distance = 1 - cosine similarity
            pos_dist = 1 - torch.sum(anchor * positive, dim=1)
            neg_dist = 1 - torch.sum(anchor * negative, dim=1)
            
        elif self.distance_metric == 'euclidean':
            # Euclidean distance
            pos_dist = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
            neg_dist = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1))
            
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Triplet loss: max(0, margin + d(a,p) - d(a,n))
        loss = F.relu(self.margin + pos_dist - neg_dist)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLossWithMetrics(TripletLoss):
    """
    Extended triplet loss that also computes useful metrics for logging.
    """
    
    def __init__(self, margin=0.3, distance_metric='cosine', reduction='mean'):
        super().__init__(margin, distance_metric, reduction)
        
    def forward(self, anchor, positive, negative, return_metrics=False):
        """
        Compute triplet loss with optional metrics.
        
        Args:
            anchor, positive, negative: Embedding tensors
            return_metrics (bool): Whether to return additional metrics
            
        Returns:
            loss (torch.Tensor): Triplet loss
            metrics (dict, optional): Additional metrics for logging
        """
        if self.distance_metric == 'cosine':
            # Normalize embeddings
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
            
            # Cosine distance = 1 - cosine similarity
            pos_dist = 1 - torch.sum(anchor * positive, dim=1)
            neg_dist = 1 - torch.sum(anchor * negative, dim=1)
            
        elif self.distance_metric == 'euclidean':
            pos_dist = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
            neg_dist = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Triplet loss
        loss_per_sample = F.relu(self.margin + pos_dist - neg_dist)
        
        if self.reduction == 'mean':
            loss = loss_per_sample.mean()
        elif self.reduction == 'sum':
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample
        
        if not return_metrics:
            return loss
        
        # Compute metrics
        with torch.no_grad():
            metrics = {
                'avg_positive_distance': pos_dist.mean().item(),
                'avg_negative_distance': neg_dist.mean().item(),
                'avg_margin_violation': loss_per_sample.mean().item(),
                'num_active_triplets': (loss_per_sample > 0).sum().item(),
                'fraction_active_triplets': (loss_per_sample > 0).float().mean().item(),
                'hardest_positive_distance': pos_dist.max().item(),
                'easiest_negative_distance': neg_dist.min().item(),
            }
        
        return loss, metrics


def test_triplet_loss():
    """Test function for triplet loss implementation."""
    print("Testing TripletLoss implementation...")
    
    # Create dummy embeddings
    batch_size, embedding_dim = 4, 512
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)
    
    # Test cosine distance
    loss_fn = TripletLossWithMetrics(margin=0.3, distance_metric='cosine')
    loss, metrics = loss_fn(anchor, positive, negative, return_metrics=True)
    
    print(f"Cosine triplet loss: {loss:.4f}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test euclidean distance
    loss_fn_euclidean = TripletLossWithMetrics(margin=0.3, distance_metric='euclidean')
    loss_euclidean = loss_fn_euclidean(anchor, positive, negative)
    
    print(f"Euclidean triplet loss: {loss_euclidean:.4f}")
    print("TripletLoss test completed successfully!")


if __name__ == '__main__':
    test_triplet_loss()