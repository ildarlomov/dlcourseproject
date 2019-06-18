import torch
from catalyst.dl.experiments import SupervisedRunner
from src.data.read_dataset import MCSDataset, FakeMCSDataset
from src.data.rnn_dataset import RNNMCSDataset, FakeRNNMCSDataset
# from src.data.baseline_transformers import ToTensor
from torch.utils.data.dataloader import DataLoader
from src.models.triplet import TripletNet, RNNTripletNet
from src.models.baseline_net import ResNetCaffe, BasicBlock
# from src.models.baseline_net_body import ResNetCaffeBody, BasicBlock
from src.models.models_for_finetuning import ResNetCaffeFinetune
from src.models.rnn import ResnetBasedRNN
from src.catalyst_hacks.triplet_runner import TripletRunner, TripletLossCallback, MCSMetricsCallback
from pathlib import Path
from functools import reduce
import torchvision as tv


def get_new_logpath(logs_path: str) -> str:
    # todo: if i deleted some dirs it will no longer work
    p = Path(logs_path)
    dirs_in_basepath = list(f for f in p.iterdir() if f.is_dir())
    return str(p / str(len(dirs_in_basepath) + 1))


if __name__ == "__main__":
    # experiment setup
    # todo: add automatic experiment naming incrementation
    base_logdir = "./models/baseline/logs"

    logdir = get_new_logpath(base_logdir)

    num_epochs = 150

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    preprocessing = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=MEAN, std=STD),
    ])

    # data
    train_ds = RNNMCSDataset(
        train_df="data/raw/train_df.csv",
        train_df_descriptors="data/raw/train_df_descriptors.npy",
        train_gt_df="data/raw/train_gt_df.csv",
        train_gt_descriptors="data/raw/train_gt_descriptors.npy",
        train_df_track_order_df="data/raw/train_df_track_order_df.csv",
        root_dir='data/raw/data',
        is_val=False,
        transform=preprocessing
    )

    dev_ds = RNNMCSDataset(
        train_df="data/raw/train_df.csv",
        train_df_descriptors="data/raw/train_df_descriptors.npy",
        train_gt_df="data/raw/train_gt_df.csv",
        train_gt_descriptors="data/raw/train_gt_descriptors.npy",
        train_df_track_order_df="data/raw/train_df_track_order_df.csv",
        root_dir='data/raw/data',
        is_val=True,
        transform=preprocessing
    )

    # todo: use maximal batch size for your gpu
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, drop_last=False)
    dev_dl = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=12, pin_memory=True, drop_last=False)

    loaders = {"train": train_dl, "dev": dev_dl}

    # model, criterion, optimizer
    # body_weights = "models/baseline/resnet_caffe_weights.pth"
    # model = ResNetCaffe([1, 2, 5, 3], BasicBlock, pretrained=True, weights_path=body_weights)
    embedding_rnn = ResnetBasedRNN(
        embeddingnet=None,  # model, if you want to optimize baseline feature extractor as well
        num_layers=1,
        dropout=0,
        hidden_size=512,
        input_size=512
    )
    tnet = RNNTripletNet(embedding_rnn)

    params_to_update = []
    print("Params to learn during transfer learning!:")
    for name, param in tnet.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)

    # Observe that all parameters are being optimized

    criterion = torch.nn.TripletMarginLoss(margin=1.0)

    optimizer = torch.optim.Adam(params_to_update)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    mcs_metrics_callback = MCSMetricsCallback()

    # model runner
    runner = TripletRunner(achor_key='gt_image', pos_key='pos_seq', neg_key='neg_seq')

    # model training
    runner.train(
        model=tnet,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=[mcs_metrics_callback],
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        valid_loader='dev',
        main_metric="mcs-TPR@FPR=1e-06",
        minimize_metric=False
    )
