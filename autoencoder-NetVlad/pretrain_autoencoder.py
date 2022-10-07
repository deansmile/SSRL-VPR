import argparse
from email.policy import default
import torch
import torch.nn as nn
import torchvision
from Autoencoder import Autoencoder
import argparse
import math
import os


nn_stack = [{"num_layer": 1, "in_channels":1, "out_channels": 8}, 
            {"num_layer": 1, "in_channels":8, "out_channels": 16, "pooling": None}]

def parse_arguments():
    parser = argparse.ArgumentParser("Pre-training CNN based autoencoder")
    parser.add_argument("--experiment_name", type=str, default="pretrain_autoencoder")
    parser.add_argument("--dataset_root_path", type=str, default="~/datasets")
    parser.add_argument("--save_checkpoint_dir", type=str)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',help='learning rate (absolute lr)')

    # Data loader parameters
    parser.add_argument("--batch_size", type=int, default=64)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_root_path, train=True, transform=transform, download=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_root_path, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = Autoencoder().to(device)
    print(model)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # mean-squared error loss
    criterion = nn.MSELoss()
    
    best_loss = math.inf
    epoch_offset = 0
    
    # loading checkpoint
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["loss"]
        epoch_offset = checkpoint["epoch"]
    
    # start training
    model.train()
    for epoch in range(epoch_offset, args.epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
           #  batch_features = batch_features.view(-1, 784).to(device)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, args.epochs, loss))

        # saving the best model
        if loss < best_loss:
            best_loss = loss
            torch.save({"epoch": epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss}, 
            os.path.join(args.save_checkpoint_dir, args.experiment_name) + ".pt")