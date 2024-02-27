import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import *
import numpy as np
import argparse
from GB_SDT import GB_SDT


def evaluate(model: GB_SDT, test_loader: DataLoader, device: torch.device):
    """
    Evaluates the GB_SDT model on a test dataset.

    Args:
        model (GB_SDT): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) to perform the evaluation on.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.predict(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f'\nTest set: Accuracy: {accuracy:.2f}%\n')


def main(args):
    """
    Main training function for GB_SDT model on MNIST.

    Args:
        args: Command-line arguments.
    """
    # Parameters from args
    n_trees = args.n_trees
    depth = args.depth
    lr = args.lr  # Learning rate for the ensemble update, not individual tree training
    internal_lr = args.internal_lr  # Learning rate for training individual trees
    lamda = args.lamda
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    epochs = args.epochs
    log_interval = args.log_interval
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        args.use_cuda = True
        print('Using CUDA')
    else:
        args.use_cuda = False
        device = torch.device('cpu')
        print('Using CPU')

    # Load datasets
    if args.dataset == 'MNIST':
        args.input_dim = 28 * 28  # MNIST images are 28x28
        args.output_dim = 10
        train_loader, val_loader, test_loader = get_mnist(
            args.data_dir, batch_size, args.output_dim)
    elif args.dataset == 'CELEBA':
        args.input_dim = 96 * 96 * 3  # CELEBA images dimensions
        args.output_dim = 2
        train_loader, val_loader, test_loader = get_celeba(
            feature_idx=args.feature_idx, data_dir=args.data_dir,
            batch_size=args.batch_size, num_train=120_000, num_test=10_000)
    elif args.dataset == 'STL_STAR':
        args.input_dim = 96 * 96 * 3  # STL_STAR image size
        args.output_dim = 2
        train_loader, val_loader, test_loader = get_stl_star(
            data_dir=args.data_dir, batch_size=args.batch_size)

    # Model initialization
    model = GB_SDT(input_dim=args.input_dim, output_dim=args.output_dim, n_trees=n_trees, lr=lr, internal_lr=internal_lr,
                   depth=depth, lamda=lamda, weight_decay=weight_decay, epochs=epochs,
                   log_interval=log_interval, use_cuda=args.use_cuda)
    # Assuming this method exists within GB_SDT
    model.train(train_loader, val_loader, test_loader)

    # Testing the model
    evaluate(model, test_loader, device=device)

    # Saving the model
    torch.save(model.state_dict(), args.save_model_path)
    print(f'Model parameters saved to {args.save_model_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GB_SDT model on MNIST.")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(),
                        'datasets'), help='Directory for storing input data')
    parser.add_argument('--dataset', type=str, choices=['MNIST',
                        'CELEBA', 'STL_STAR'], default='MNIST', help='Dataset to use.')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index for CelebA dataset (only relevant for CelebA)')
    parser.add_argument('--n_trees', type=int, default=4,
                        help='Number of trees in the ensemble.')
    parser.add_argument('--depth', type=int, default=5,
                        help='Depth of each tree.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for ensemble update.')
    parser.add_argument('--internal_lr', type=float, default=0.001,
                        help='Learning rate for individual trees.')
    parser.add_argument('--lamda', type=float, default=1e-3,
                        help='Lambda for regularization.')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='Weight decay for optimization.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging during training.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Whether to shuffle the dataset.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--save_model_path', type=str,
                        default='GB_SDT_celeba_smiling.pth', help='Path to save the trained model.')
    parser.add_argument('--use_cuda', action='store_true',
                        default=False, help='Enable CUDA if available.')
    args = parser.parse_args()
    main(args)
