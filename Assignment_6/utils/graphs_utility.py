import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_accuracy_curves(train_loss_values, test_loss_values, train_accuracy, test_accuracy):
    sns.set(style='whitegrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (20, 10)

    "group_norm": ["GroupNorm", False, False],
    "layer_norm": ["LayerNorm", False, False],
    "batch_norm_l1": ["BatchNorm", True, False],
    "group_norm_l1": ["GroupNorm", True, False],
    "layer_norm_l1": ["LayerNorm", True, False],
    "batch_norm_l1_l2": ["BatchNorm", True, True]


    # Plot the learning curve.
    fig, (plt1, plt2) = plt.subplots(1, 2)
    plt1.plot(np.array(train_loss_values[0]), 'r', label="Group Norm Training Loss")
    plt1.plot(np.array(train_loss_values[0]), 'g', label="Layer Norm Training Loss")
    plt1.plot(np.array(train_loss_values[0]), 'b', label="Batch Norm + L1 Training Loss")
    plt1.plot(np.array(train_loss_values[0]), 'c', label="Group Norm + L1 Training Loss")
    plt1.plot(np.array(train_loss_values[0]), 'y', label="Layer Norm + L1 Training Loss")
    plt1.plot(np.array(train_loss_values[0]), 'm', label="Batch Norm + L1 + L2 Training Loss")



    plt1.plot(np.array(test_loss_values), 'b', label="Validation Loss")
    plt2.plot(np.array(train_accuracy_values), 'r', label="Training Accuracy")
    plt2.plot(np.array(test_accuracy_values), 'b', label="Validation Accuracy")

    plt2.set_title("Training-Validation Accuracy Curve")
    plt2.set_xlabel("Epoch")
    plt2.set_ylabel("Accuracy")
    plt2.legend()
    plt1.set_title("Training-Validation Loss Curve")
    plt1.set_xlabel("Epoch")
    plt1.set_ylabel("Loss")
    plt1.legend()

    plt.show()