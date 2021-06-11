"""

"""
import torch.cuda


def get_model_summary(model):
    """
    Summary of model will be printed
    Args:
        model: Model

    Returns: summary of model

    """
    from torchsummary import summary
    model_summary = summary(model, input_size=(1, 28, 28))
    return model_summary


def load_optimizer(model):
    """
    define optimizer and scheduler
    Args:
        model: model

    Returns: optimizer and scheduler

    """
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.7)

    return optimizer


def get_cuda():
    seed = 1
    cuda = torch.cuda.is_available()
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    return cuda




