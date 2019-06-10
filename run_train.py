import torch
from catalyst.dl.experiments import SupervisedRunner
from src.data.read_dataset import MCSDataset, FakeMCSDataset
from src.data.baseline_transformers import ToTensor
from torch.utils.data.dataloader import DataLoader
from src.models.triplet import TripletNet
from src.models.baseline_net import ResNetCaffe, BasicBlock
from src.catalyst_hacks.triplet_runner import TripletRunner, TripletLossCallback

if __name__ == "__main__":
    # experiment setup
    # todo: add automatic experiment naming incrementation
    logdir = "./models/baseline/logs/exp1"
    num_epochs = 42

    # data
    train_ds = MCSDataset(tracks_df_csv='data/raw/train_df.csv',
                          order_df_csv='data/raw/train_df_track_order_df.csv',
                          gt_csv='data/raw/train_gt_df.csv',
                          root_dir='data/raw/data',
                          is_val=False,
                          transform=ToTensor())

    dev_ds = MCSDataset(tracks_df_csv='data/raw/train_df.csv',
                        order_df_csv='data/raw/train_df_track_order_df.csv',
                        gt_csv='data/raw/train_gt_df.csv',
                        root_dir='data/raw/data',
                        is_val=True,
                        transform=ToTensor())

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4)

    loaders = {"train": train_dl, "dev": dev_dl}

    # model, criterion, optimizer
    # todo: we have an option to use pretrained weights
    model = ResNetCaffe([1, 2, 5, 3], BasicBlock, pretrained=False)
    tnet = TripletNet(model)
    criterion = torch.nn.MarginRankingLoss(margin=1.0)
    optimizer = torch.optim.Adam(tnet.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # model runner
    runner = TripletRunner(achor_key='track_image', pos_key='pos_image', neg_key='neg_image')
    loss_callback = TripletLossCallback(loss_key='loss')

    # model training
    runner.train(
        model=tnet,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=[loss_callback],
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        valid_loader='dev'
    )
