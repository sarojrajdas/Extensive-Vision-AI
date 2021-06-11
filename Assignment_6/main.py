import torch

from utils.data_utility import download_train_data, download_test_data, \
    train_data_transformation, test_data_transformation, \
    load_train_data, load_test_data, get_data_loader_args
from utils.misc import get_cuda, get_model_summary, load_optimizer

from models.batch_norm import BatchNormModel
from models.group_norm import GroupNormModel
from models.layer_norm import LayerNormModel

from utils import train_utility, test_utiliy

MODELS = {
    "GroupNorm": GroupNormModel(),
    "LayerNorm": LayerNormModel(),
    "BatchNorm": BatchNormModel()
}


def run():
    train_transforms = train_data_transformation()
    test_transofrms = test_data_transformation()

    train_data = download_train_data(train_transforms=train_transforms)
    test_data = download_test_data(test_transforms=test_transofrms)

    cuda = get_cuda()
    data_loader_args = get_data_loader_args(cuda)

    train_loader = load_train_data(train_data, **data_loader_args)
    test_loader = load_test_data(test_data, **data_loader_args)

    models = {
        "group_norm": ["GroupNorm", False, False],
        "layer_norm": ["LayerNorm", False, False],
        "batch_norm_l1": ["BatchNorm", True, False],
        "group_norm_l1": ["GroupNorm", True, False],
        "layer_norm_l1": ["LayerNorm", True, False],
        "batch_norm_l1_l2": ["BatchNorm", True, True]
    }

    device = torch.device("cuda" if cuda else "cpu")

    final_train_loss_values = []
    final_test_loss_values = []
    final_train_accuracy_values = []
    final_test_accuracy_values = []

    for key, value in models.items():
        print(f"=============== Running {key} Model ===============")
        train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values = run_epochs(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model=value[0],
            is_l1_norm=value[1],
            is_l2_norm=value[2],
            tag= key
        )

        final_train_loss_values.append(train_loss_values)
        final_test_loss_values.append(test_loss_values)
        final_train_accuracy_values.append(train_accuracy_values)
        final_test_accuracy_values.append(test_accuracy_values)
        print(f"=============== Finished {key} Model ===============")
    # summary = get_model_summary(model=model)
    print(final_train_loss_values)
    print(final_test_loss_values)
    print(final_train_accuracy_values)
    print(final_test_accuracy_values)


def run_epochs(train_loader, test_loader, device, model, is_l1_norm, is_l2_norm, tag):
    from torch.optim.lr_scheduler import StepLR, OneCycleLR

    model = MODELS[model].to(device) # Get Model Instance
    summary = get_model_summary(model=model)
    print(summary)

    optimizer = load_optimizer(model)
    scheduler = OneCycleLR(optimizer, max_lr=0.015, epochs=20, steps_per_epoch=len(train_loader))

    train_loss_values = []
    test_loss_values = []
    train_accuracy_values = []
    test_accuracy_values = []

    for epoch in range(1, 21):
        train_acc, train_loss = train_utility.train(
            model=model,
            device=device,
            train_loader=train_loader,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            is_l1_norm=is_l1_norm,
            is_l2_norm=is_l2_norm
        )

        test_acc, test_loss = test_utiliy.test(
            model=model,
            device=device,
            test_loader=test_loader
        )

        train_accuracy_values.append(train_acc)
        train_loss_values.append(train_loss)

        test_accuracy_values.append(test_acc)
        test_loss_values.append(test_loss)

    return train_loss_values, test_loss_values, train_accuracy_values, test_accuracy_values


if __name__ == "__main__":
    run()
