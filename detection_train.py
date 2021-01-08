from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from detection import *
from utils import CellImageLoad, CellImageLoadTest, CellImageLoadPseudo, set_seed
import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
from networks import UNet, MaskMSELoss
import argparse

seed = 1


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
    random.seed(worker_id + seed)
    np.random.seed(worker_id + seed)


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        help="dataset",
        default="c2c12",
        type=str,
    ),
    parser.add_argument(
        "-v", "--validation", dest="val", help="validation", default=True, type=bool
    ),
    parser.add_argument(
        "-cv", "--cv_num", dest="cv_n", help="cross validation numver", default=0, type=int
    )
    parser.add_argument(
        "-c", "--channel", dest="channel", help="img channel 1 or 3", default=1, type=int
    )
    parser.add_argument(
        "-ctcd", "--ctc_dataset", dest="ctcd", help="dataset_number", default=1, type=int
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true", default=True
    )
    parser.add_argument(
        "-dn", "--device", dest="device", help="select gpu device", default=0, type=int
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=8, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=500, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--decay", dest="decay", help="weight decay", default=0.5, type=float
    )
    parser.add_argument(
        "--crop_size", dest="crop_size", help="crop size", default=(128, 128), type=tuple
    )
    parser.add_argument(
        "--visdom", dest="vis", help="visdom show", default=True, type=bool
    )
    parser.add_argument(
        "--patience", dest="patience", help="set patience", default=500, type=int
    )
    parser.add_argument(
        "--pseudo", dest="pseudo", help="pseudo flag", default=False, type=bool
    )
    parser.add_argument(
        "--pre", dest="pre", help="pre train", default=False, type=bool
    )

    args = parser.parse_args()
    return args


def gather_path(train_paths, mode, extension):
    ori_paths = []
    for train_path in train_paths:
        ori_paths.extend(sorted(train_path.joinpath(mode).glob(extension)))
    return ori_paths


class _TrainBase:
    def __init__(self, args):
        if args.pseudo:
            data_loader = CellImageLoadPseudo(args.imgs, args.likelis, args.bg_mask, args.dataset, args.channel,
                                              args.crop_size)
        else:
            data_loader = CellImageLoad(args.imgs, args.likelis, args.dataset, args.channel, args.crop_size)

        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init_fn
        )
        self.trainconfloader = data_loader

        self.val_dataloader = CellImageLoadTest(args.imgs, args.dataset, args.channel, args.crop_size)

        self.number_of_traindata = data_loader.__len__()

        self.val = args.val

        if self.val:
            data_loader = CellImageLoad(args.val_imgs, args.val_likelis, args.dataset, args.channel, args.crop_size)

            self.val_loader = torch.utils.data.DataLoader(
                data_loader, batch_size=args.batch_size, shuffle=False, num_workers=0, worker_init_fn=worker_init_fn
            )

        self.save_weight_path = args.weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                args.epochs, args.batch_size, args.learning_rate, args.gpu
            )
        )

        self.net = args.net

        self.train = None
        self.val = args.val

        self.vis = args.vis
        self.dataset = args.dataset
        self.N_train = None
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.decay = args.decay
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        self.device = args.device
        if args.pseudo:
            self.criterion = MaskMSELoss()
        else:
            self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0
        self.patience = args.patience
        self.pseudo = args.pseudo

    def show_graph(self):
        x = list(range(len(self.losses)))
        plt.plot(x, self.losses)
        # plt.plot(x, self.val_losses)
        plt.show()


class TrainNet(_TrainBase):

    def create_vis_show(self):
        return self.vis.images(
            torch.ones((self.batch_size, 1, 256, 256)), self.batch_size
        )

    def update_vis_show(self, images, window1):
        self.vis.images(images, self.batch_size, win=window1)

    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1)).cpu(),
            opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
        )

    def update_vis_plot(self, iteration, loss, window1, update_type):
        self.vis.line(
            X=torch.ones((1)).cpu() * iteration,
            Y=torch.Tensor(loss).unsqueeze(0).cpu(),
            win=window1,
            update=update_type,
        )

    def main(self):
        if self.vis:
            import visdom

            HOSTNAME = "localhost"
            PORT = 8097

            self.vis = visdom.Visdom(port=PORT, server=HOSTNAME, env=self.dataset)

            vis_title = "ctc"
            vis_legend = ["Loss"]
            vis_epoch_legend = ["Loss", "Val Loss"]

            self.iter_plot = self.create_vis_plot(
                "Iteration", "Loss", vis_title, vis_legend
            )
            self.epoch_plot = self.create_vis_plot(
                "Epoch", "Loss", vis_title, vis_epoch_legend
            )
            self.ori_view = self.create_vis_show()
            self.gt_view = self.create_vis_show()
            self.pred_view = self.create_vis_show()
            self.bg_mask_view = self.create_vis_show()

        iteration = 0
        for epoch in range(self.epochs):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))

            self.net.train()

            pbar = tqdm(total=self.number_of_traindata)
            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]
                if self.pseudo:
                    bg_mask = data["bg"]
                if self.gpu:
                    imgs = imgs.cuda(self.device)
                    true_masks = true_masks.cuda(self.device)
                    if self.pseudo:
                        bg_mask = bg_mask.cuda(self.device)

                mask_preds = self.net(imgs)

                if self.pseudo:
                    loss = self.criterion(mask_preds, true_masks, bg_mask)
                else:
                    loss = self.criterion(mask_preds, true_masks)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iteration += 1
                if self.vis:
                    self.update_vis_plot(
                        iteration, [loss.item()], self.iter_plot, "append"
                    )
                    mask_preds = (mask_preds - mask_preds.min()) / (mask_preds.max() - mask_preds.min())
                    self.update_vis_show(imgs.cpu(), self.ori_view)
                    self.update_vis_show(mask_preds, self.pred_view)
                    self.update_vis_show(true_masks.cpu(), self.gt_view)
                    if self.pseudo:
                        self.update_vis_show(bg_mask.cpu(), self.bg_mask_view)
                if (iteration > 10000) and not self.val:
                    break

                pbar.update(self.batch_size)

            pbar.close()
            loss = self.epoch_loss / (self.number_of_traindata + 1)
            print("Epoch finished ! Loss: {}".format(loss))
            self.losses.append(loss)
            self.epoch_loss = 0

            if epoch % 10 == 0:
                torch.save(
                    self.net.state_dict(),
                    str(
                        self.save_weight_path.parent.joinpath(
                            "epoch_weight/{:05d}.pth".format(epoch)
                        )
                    ),
                )

            if self.val:
                self.validation(loss, epoch)

            if (iteration > 10000) and not self.val:
                print("stop running")
                torch.save(self.net.state_dict(), str(self.save_weight_path))
                break

        torch.save(self.net.state_dict(), str(self.save_weight_path))
        self.show_graph()

    def validation(self, loss, epoch):
        val_loss = eval_net(self.net, self.val_loader, gpu=self.gpu)
        if loss < 0.1:
            print("val_loss: {}".format(val_loss))
            try:
                if min(self.val_losses) > val_loss:
                    print("update best")
                    torch.save(self.net.state_dict(), str(self.save_weight_path))
                    self.bad = 0
                else:
                    self.bad += 1
                    print("bad ++")
            except ValueError:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
            self.val_losses.append(val_loss)
        else:
            print("loss is too large. Continue train")
            self.val_losses.append(val_loss)
        if self.vis:
            self.update_vis_plot(
                iteration=epoch,
                loss=[loss, val_loss],
                window1=self.epoch_plot,
                update_type="append",
            )
        print("bad = {}".format(self.bad))
        self.epoch_loss = 0


def call(args):
    net = UNet(n_channels=args.channel, n_classes=1)
    if args.pre:
        net.load_state_dict(torch.load(f"weight/super/{args.dataset}/{args.seq:02d}/best.pth", map_location="cpu"))
    if args.gpu:
        net.cuda(args.device)

    args.net = net

    train = TrainNet(args)

    train.main()


def supervised(args):
    args.vis = False

    for args.dataset in MODES.values():
        args.dataset = args.dataset
        base_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{args.dataset}")
        for args.seq in [1, 2]:
            if args.dataset == "PhC-C2DL-PSC":
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}-S").glob("*.png"))[150:250]
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK-S").glob("*.tif"))
            elif args.dataset in SCALED_DATASET:
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}-S").glob("*.png"))
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK-S").glob("*.tif"))
            else:
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}").glob("*.tif"))
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK").glob("*.tif"))

            args.weight_path = Path(f"./weight/super/{args.dataset}/{args.seq:02d}/best.pth")

            args.val = False
            args.epochs = 1000
            call(args)


def pseudo_test(args):
    for args.dataset in MODES.values():
        unsup_base_path = Path(f"./output/select_pseudo/test/{args.dataset}/3-3")
        for args.seq, un_seq in [[1, 2], [2, 1]]:
            skip_image = int(unsup_base_path.name[0]) // 2
            args.imgs = sorted(unsup_base_path.joinpath(f"{un_seq:02d}/img").glob("*.tif"))[
                        skip_image:-skip_image]
            args.likelis = sorted(unsup_base_path.joinpath(f"{un_seq:02d}/fg_pseudo").glob("*.tif"))[
                           skip_image:-skip_image]
            args.bg_mask = sorted(unsup_base_path.joinpath(f"{un_seq:02d}/bg_pseudo").glob("*.tif"))

            args.vis = True
            args.val = False
            args.pseudo = True
            args.epochs = 1000
            args.pre = True
            args.weight_path = Path(f"./weight/pseudo_pre/{args.dataset}/{un_seq:02d}/best.pth")
            call(args)


def unlabel_direct(args):
    for args.dataset in MODES.values():
        args.dataset = args.dataset
        base_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{args.dataset}")
        unsup_base_path = Path(
            f"/home/kazuya/main/semisupervised_detection/output/detection/unsupervised_pred/{args.dataset}")
        for args.seq in [1, 2]:
            if args.dataset == "PhC-C2DL-PSC":
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}-S").glob("*.png"))[150:250]
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK-S").glob("*.tif"))
            elif args.dataset in SCALED_DATASET:
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}-S").glob("*.png"))
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK-S").glob("*.tif"))
            else:
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}").glob("*.tif"))
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK").glob("*.tif"))

            for un_seq in [1, 2]:
                args.imgs.extend(sorted(unsup_base_path.joinpath(f"{args.seq:02d}-{un_seq:02d}/ori").glob("*.tif")))
                args.likelis.extend(sorted(unsup_base_path.joinpath(f"{args.seq:02d}-{un_seq:02d}/pred").glob("*.tif")))
            args.weight_path = Path(f"./weight/unsup_direct/{args.dataset}/{args.seq:02d}/best.pth")

            args.val = False
            args.epochs = 10000
            call(args)


def unpseudolabel_train(args):
    for args.dataset in MODES.values():
        args.dataset = args.dataset
        base_path = Path(f"/home/kazuya/dataset/Cell_tracking_challenge/{args.dataset}")
        unsup_base_path = Path(f"./output/select_pseudo/{args.dataset}/3-3")
        for args.seq in [1, 2]:
            if args.dataset == "PhC-C2DL-PSC":
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}-S").glob("*.png"))[150:250]
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK-S").glob("*.tif"))
                args.bg_mask = [0 for i in range(len(args.imgs))]
            elif args.dataset in SCALED_DATASET:
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}-S").glob("*.png"))
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK-S").glob("*.tif"))
                args.bg_mask = [0 for i in range(len(args.imgs))]
            else:
                args.imgs = sorted(base_path.joinpath(f"{args.seq:02d}").glob("*.tif"))
                args.likelis = sorted(base_path.joinpath(f"{args.seq:02d}_GT/LIK").glob("*.tif"))
                args.bg_mask = [0 for i in range(len(args.imgs))]

            for un_seq in [1, 2]:
                skip_image = int(unsup_base_path.name[0]) // 2
                args.imgs.extend(sorted(unsup_base_path.joinpath(f"{args.seq:02d}-{un_seq:02d}/img").glob("*.tif"))[
                                 skip_image:-skip_image])
                args.likelis.extend(
                    sorted(unsup_base_path.joinpath(f"{args.seq:02d}-{un_seq:02d}/fg_pseudo").glob("*.tif"))[
                    skip_image:-skip_image])
                args.bg_mask.extend(
                    sorted(unsup_base_path.joinpath(f"{args.seq:02d}-{un_seq:02d}/bg_pseudo").glob("*.tif")))

            assert len(args.imgs) == len(args.likelis), "the number of images is wrong"
            assert len(args.bg_mask) == len(args.likelis), "the number of images is wrong"
            args.weight_path = Path(f"./weight/unpseudo/{args.dataset}/{args.seq:02d}/best.pth")

            args.pseudo = True
            args.val = False
            args.epochs = 1000
            call(args)


MODES = {
    # 1: "BF-C2DL-HSC",
    # 2: "BF-C2DL-MuSC",
    # 3: "DIC-C2DH-HeLa",
    4: "PhC-C2DH-U373",
    # 6: "Fluo-C2DL-MSC",
    7: "Fluo-N2DH-GOWT1",
    8: "Fluo-N2DH-SIM+",
    9: "Fluo-N2DL-HeLa",
}

SCALED_DATASET = [
    "DIC-C2DH-HeLa",
    "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1",
    "Fluo-N2DH-SIM+",
    "Fluo-N2DL-HeLa",
    "PhC-C2DH-U373",
    "PhC-C2DL-PSC",
]

CTC_config = {"DIC-C2DH-HeLa": "s_scale_norm",
              "Fluo-C2DL-MSC": "l_scale_norm",
              "Fluo-N2DH-GOWT1": "s_scale_norm",
              "Fluo-N2DH-SIM+": "ori_norm",
              "Fluo-N2DL-HeLa": "s_scale",
              "PhC-C2DH-U373": "l_scale",
              "PhC-C2DL-PSC": "ori"}

if __name__ == "__main__":
    set_seed(seed)
    args = parse_args()
    # supervised(args)
    pseudo_test(args)
    # unlabel_direct(args)
    # unpseudolabel_train(args)
