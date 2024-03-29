import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    pre_residual_model = nn.Sequential(
        nn.Linear(dim, hidden_dim), 
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob), # be carefule the drop out !!
        nn.Linear(hidden_dim, dim), 
        norm(dim)
        )
    model = nn.Sequential(
        nn.Residual(pre_residual_model),
        nn.ReLU()
        )
    return model 
    
    
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    pre_model = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    )
    residual_blocks = nn.Sequential(
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)]
    )
    post_model = nn.Linear(hidden_dim, num_classes)
    return nn.Sequential(
        pre_model, 
        residual_blocks,
        post_model
    )
   
    ### END YOUR SOLUTION



def epoch(dataloader, model, opt : ndl.optim.Optimizer = None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    correct, loss_sum, n_step, n_samples= 0., 0., 0, 0
    if opt:
        model.train()
    else:
        model.eval()
    for X, y in dataloader:
        if opt:
            opt.reset_grad()
        pred = model(X)
        loss = loss_func(pred, y)
        correct += (pred.numpy().argmax(axis=1) == y.numpy()).sum()
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.numpy()
        n_step += 1
        n_samples += X.shape[0]
    
    # NOTE (1 - mean of acc over iterations) is not accurate enough
    return (1 - correct / n_samples), loss_sum / n_step
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    import os 
    train_data = ndl.data.MNISTDataset(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_data = ndl.data.MNISTDataset(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    train_dataloader = ndl.data.DataLoader(train_data, batch_size)
    test_dataloader = ndl.data.DataLoader(test_data, batch_size)
    model = MLPResNet(28 * 28, hidden_dim)
    optim = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_acc, train_loss = epoch(train_dataloader, model, optim)
        test_acc, test_loss = epoch(test_dataloader, model)
    return train_acc, train_loss, test_acc, test_loss
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
