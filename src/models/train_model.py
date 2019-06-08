import torch
from catalyst.dl.experiments import SupervisedRunner
from resnet_cafee import ResNetCaffe, BasicBlock

# experiment setup
logdir = "./logdir"
num_epochs = 42

# data
loaders = {"train": ..., "valid": ...}

# model, criterion, optimizer
model = ResNetCaffe([1, 2, 5, 3], BasicBlock, pretrained=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)