from torchvision import transforms

def train_data_transformation():
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                        #transforms.ToPILImage(),
                                        #transforms.Resize((28, 28)),
                                        #transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        #transforms.RandomRotation((-7.0, 7.0), fill=(0.13,)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                        # Note the difference between (0.1307) and (0.1307,)
                                        #transforms.RandomRotation((-5.0, 5.0), fill=(-0.42,)),
                                        transforms.RandomAffine((-5,5), translate=None, scale=None, shear=5, resample=False, fillcolor=(-0.42,))
                                        ])
    return train_transforms

def test_data_transformation():
    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
    return test_transforms