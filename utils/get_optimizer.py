import os
import torch.optim as optim
from .schedulefree.sgd_schedulefree import SGDScheduleFree
from .get_config_value import get_config_value

def get_optimizer(param, config):
    """
    Returns an optimizer instance based on the environment variable 'OPTIMIZER'.
    If 'OPTIMIZER' is not set, it defaults to 'adam'.
    """
    #optimizer_name = os.getenv('OPTIMIZER', 'adam').lower()
    optimizer_name = get_config_value(config, "optimizer")

    if optimizer_name == 'adam':
        return optim.Adam(param, get_config_value(config, "init_lr"), betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer_name == 'sgd':
        return  optim.SGD(param, get_config_value(config, "init_lr"), momentum=0.9, weight_decay=0)
    elif optimizer_name == 'adamw':
        return optim.AdamW(param, get_config_value(config, "init_lr"), betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(param, get_config_value(config, "init_lr"), alpha=0.99, eps=1e-08, weight_decay=0)
    elif optimizer_name == 'adagrad':
        return optim.Adagrad(param, get_config_value(config, "init_lr"), lr_decay=0, weight_decay=0)
    elif optimizer_name == 'sgd_schedulefree':
        return SGDScheduleFree(param, get_config_value(config, "init_lr"))
    elif optimizer_name == 'adam_schedulefree':
        return schedulefree.adamw_schedulefree(param, get_config_value(config, "init_lr"))
    elif optimizer_name == 'radam_schedulefree':
        return schedulefree.radam_schedulefree(param, get_config_value(config, "init_lr"))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")