import torch
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score

def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]

def mean_precision(y, y_pred):
    """Calculates unweighted mean precision.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    mean_precision = precision_score(y, y_pred, average='macro')  
    return mean_precision

def mean_recall(y, y_pred):
    """Calculates unweighted mean recall.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    mean_recall = recall_score(y, y_pred, average='macro')  
    return mean_recall

def cohen_kappa(y, y_pred):
    """Calculates unweighted mean recall.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    cohen_kappa = cohen_kappa_score(y, y_pred)  
    return cohen_kappa

NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy,
    'mean_precision': mean_precision,
    'mean_recall': mean_recall,
    'cohen_kappa': cohen_kappa
}
