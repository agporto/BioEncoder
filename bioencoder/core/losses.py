import torch.nn as nn
import torch
from pytorch_metric_learning import losses


class SupConLoss(nn.Module):
    """
    Computes the Supervised Contrastive Loss as described in the paper
    "Supervised Contrastive Learning" (https://arxiv.org/pdf/2004.11362.pdf)
    and supports the unsupervised contrastive loss in SimCLR.

    The contrastive loss encourages the embeddings to be close to their positive
    samples and far away from negative samples. It measures the similarity between
    two samples by the dot product of their embeddings and apply a temperature
    scaling.

    Args:
        temperature (float, optional): The temperature scaling.Default: `0.07`.
        contrast_mode (str, optional): Specifies the mode to compute contrastive loss.
            There are two modes: `all` and `one`. In `all` mode, every sample is used
            as an anchor. In `one` mode, only the first is used as an anchor.
            Default: `'all'`.
        base_temperature (float, optional): The base temperature used to normalize the
            temperature. Default: `0.07`.
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if base_temperature <= 0:
            raise ValueError("base_temperature must be > 0")
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - torch.log(exp_logits_sum)

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        valid_anchor_mask = mask_sum > 0
        mean_log_prob_pos = torch.zeros_like(mask_sum)
        mean_log_prob_pos[valid_anchor_mask] = (
            (mask * log_prob).sum(1)[valid_anchor_mask] / mask_sum[valid_anchor_mask]
        )

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        if not torch.any(valid_anchor_mask):
            return logits.sum() * 0.0
        loss = loss[valid_anchor_mask].mean()

        return loss


class LabelSmoothingLoss(nn.Module):
    """
    Implements the Label Smoothing Loss for classification problems.

    Args:
    - classes (int): The number of classes in the classification problem.
    - smoothing (float, optional): The smoothing factor for the target distribution. 
        The default value is 0.
    - dim (int, optional): The dimension along which the loss should be computed. 
        The default value is -1.

    Methods:
    - forward(pred, target): Computes the label smoothing loss between `pred` 
        and `target` tensors.

    """
    def __init__(self, classes, smoothing=0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        if not isinstance(classes, int) or classes <= 1:
            raise ValueError("classes must be an integer > 1")
        if not (0 <= smoothing < 1):
            raise ValueError("smoothing must be in the range [0, 1)")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("Inputs must be tensors")
        if pred.ndim < 2:
            raise ValueError("pred must be at least 2-dimensional [batch, classes, ...]")
        if pred.shape[0] != target.shape[0]:
            raise ValueError("Input tensors must have the same batch size")
        class_dim = self.dim if self.dim >= 0 else pred.ndim + self.dim
        if pred.shape[class_dim] != self.cls:
            raise ValueError(
                f"pred class dimension ({pred.shape[class_dim]}) does not match configured classes ({self.cls})"
            )
        if not torch.is_floating_point(pred):
            raise TypeError("pred must be a floating-point tensor")
        if target.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("target must be an integer tensor")
        if target.numel() > 0:
            if target.min() < 0 or target.max() >= self.cls:
                raise ValueError("target contains class indices outside valid range [0, classes-1]")
            
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


LOSSES = {
    "SupCon": SupConLoss,
    "LabelSmoothing": LabelSmoothingLoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    'SubCenterArcFace': losses.SubCenterArcFaceLoss,
    'ArcFace': losses.ArcFaceLoss,
}
