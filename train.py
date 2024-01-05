import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0" 
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YoloLoss
from torch.utils.tensorboard import SummaryWriter

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0.1
EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "checkpoints/yolo_voc_10k.pth.tar"
IMG_DIR = "data/images"

# For the dataset

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        
        return img, bboxes

# add normalization
transforms = Compose(
    [
        transforms.Resize((448, 448)), transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    overall_mean_loss = sum(mean_loss)/len(mean_loss)

    print(f"Mean loss was {overall_mean_loss}")

    return overall_mean_loss

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/train_10k.csv",  # 100examples.csv for quick checks
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir="data/labels/",
    )

    val_dataset = VOCDataset(
        "data/val.csv",
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir="data/labels/",
    )

    test_dataset = VOCDataset(
        "data/test.csv",
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir="data/labels/",
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    best_val_map = 0
    patience = 0
    writer = SummaryWriter(f"logs/VOC_10k")
    step = 0
    global_t = 0
    for epoch in range(EPOCHS):
        t_start = time.time()
        print(f"\n*** Epoch {epoch} ***")
        # train
        loss_train = train_fn(train_loader, model, optimizer, loss_fn)
        scheduler.step(loss_train)

        # get train pred results
        pred_boxes_train, target_boxes_train = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec_train = mean_average_precision(
            pred_boxes_train, target_boxes_train, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec_train}")
        t_end = time.time()
        global_t += t_end - t_start

        # add logs in tensorboard
        writer.add_scalar("Training Loss", loss_train, global_step=step)
        writer.add_scalar("Training mAP", mean_avg_prec_train, global_step=step)
        step += 1

        # perform validation for every 5th epoch (after first 10)
        if epoch % 5 == 0 and epoch > 10:
            # get val pred results
            pred_boxes_val, target_boxes_val = get_bboxes(
                val_loader, model, iou_threshold=0.5, threshold=0.4
            )
            mean_avg_prec_val = mean_average_precision(
                pred_boxes_val, target_boxes_val, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Validation mAP: {mean_avg_prec_val}")
            writer.add_scalar("Validation mAP", mean_avg_prec_val, global_step=step)

            # save model if it was better than previous validation
            if mean_avg_prec_val > best_val_map:
                best_val_map = mean_avg_prec_val
                patience = 0
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            else:
                patience += 1
            
        # perform early stopping if no improvements for 15 epochs
        if patience >= 3:
            print("\n***Early stopping executed at epoch {epoch} due to no improvements in validation mAP for the last 15 epochs***\n")
            break

    print(f"\nTraining completed, total training time: {round(global_t / 3600, 2)} hrs\n")
                


if __name__ == "__main__":
    main()