from __future__ import annotations
from abc import ABC, abstractmethod

import torch.cuda
from torchsummary import summary

class Creator(ABC):
    @abstractmethod
    def create_model(self):
        pass

    def run(self):
        from pattern.utils.data_utility import DataUtility
        train_transofrms = DataUtility.train_data_transformation()
        test_transforms = DataUtility.test_data_transformation()

        train_data = DataUtility.download_train_data(train_transforms=train_transofrms)
        test_data = DataUtility.download_test_data(test_transforms=test_transforms)

        SEED = 1

        # CUDA?
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)

        # For reproducibility
        torch.manual_seed(SEED)

        if cuda:
            torch.cuda.manual_seed(SEED)

        # data loader arguments - something you'll fetch these from cmdprmt
        data_loader_args = dict(shuffle=True, batch_size=128, num_workers=1, pin_memory=True) if cuda else dict(
            shuffle=True, batch_size=64)

        model = self.create_model()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)
        model = model.to(device)
        summary(model, input_size=(1, 28, 28))



class BatchNormModelCreator(Creator):
    def create_model(self):
        from pattern.models.BathNormModel import BatchNormModel
        return BatchNormModel()

class GroupNormModelCreator(Creator):
    def create_model(self):
        from pattern.models.GroupNormModel import GroupNormModel
        return GroupNormModel()

def client_code(creator: Creator):
    creator.run()

if __name__ == "__main__":
    client_code(GroupNormModelCreator())