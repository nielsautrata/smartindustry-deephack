import argparse
import os

import torch
import yaml
from ignite.contrib import metrics

import constants as const
import dataset
import fastflow
import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
        is_inference=True
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def inference(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataloader = build_test_data_loader(args, config)
    if torch.cuda.is_available(): model.cuda()
    model.eval()
    if args.auc: auroc_metric = metrics.ROC_AUC()
    tmp_idx = 0
    for data, targets, raws in dataloader:
        if torch.cuda.is_available(): data = data.cuda()
        with torch.no_grad():
            ret = model(data)
            outputs = ret["anomaly_map"].cpu().detach()
            for idx in range(outputs.size(0)):
                img  = cv.cvtColor(raws[idx].numpy(), cv.COLOR_RGB2BGR)
                if isinstance(args.resize, tuple):
                    img = cv.resize(img, args.resize, cv.INTER_AREA)

                pred = process(outputs[idx], inv=True, thresh=0.9, resize=img.shape[:2][::-1])
                gt   = process(targets[idx], resize=img.shape[:2][::-1])
                comb = img.astype(float)*0.7 + pred.astype(float)*0.3
                out_viz = np.concatenate([img, pred, comb, gt], axis=1)
                cv.imwrite("%s/%d.jpg"%(args.save_dir, tmp_idx), out_viz)
                tmp_idx+=1
            if args.auc:
                outputs = outputs.flatten()
                targets = targets.flatten()
                auroc_metric.update((outputs, targets))
    if args.auc:
        auroc = auroc_metric.compute()
        print("AUROC: {}".format(auroc))

def process(torch_tensor, inv=False, thresh=None, resize=None):
    prob = torch_tensor.permute(1,2,0).numpy()

    if inv: # TODO check convert probability
        prob = 1 + prob
    if thresh is not None:
        prob[prob<=thresh]=0   
    
    viz = (prob*255).astype(np.uint8)[:,:,0]
    
    if resize is not None:
        viz = cv.resize(viz, resize, cv.INTER_AREA)
    return cv.applyColorMap(viz, cv.COLORMAP_JET)

def save_fig(img, idx):
    plt.imshow(img)
    plt.savefig("%d.jpg"%idx)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to dataset folder")
    parser.add_argument("-cat", "--category", type=str, required=True, help="category of dataset Ex: capsule, screw, table")


    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, required=True, help="path to load checkpoint"
    )

    parser.add_argument(
        "--auc", action="store_true", help="Calculate AUC"
    )

    parser.add_argument(
        "--save_dir",type=str, default=None, help="result/dir_name to save visualized data"
    )

    parser.add_argument(
        "--resize",type=str, default="(255,255)", help="size of output img"
    )

    args = parser.parse_args()
    args.resize = eval(args.resize)
    return args


if __name__ == "__main__":
    import os

    def create_dirname_by_time():
        import datetime
        now = datetime.datetime.now()
        save_dir = "result/%.2d%.2d%.2d_%.2d%.2d%.2d"%(now.year, now.month, now.day, now.hour, now.minute, now.second)
        os.mkdir(save_dir)
        return save_dir

    args = parse_args()
    if not os.path.isdir("result"):
        os.mkdir("result")
    if args.save_dir is None:
        args.save_dir = create_dirname_by_time()
    elif os.path.isdir("result/"+args.save_dir):
        print(" Directory: %s existed!"%args.save_dir)
        args.save_dir = create_dirname_by_time()
    else:
        os.mkdir(args.save_dir)

    print(args)
    inference(args)
    print(" Data saved to %s"%args.save_dir)

