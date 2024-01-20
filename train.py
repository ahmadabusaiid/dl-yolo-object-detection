import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0" 
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from model import Yolov1, YOLOv1ResNet, YOLO_V1_HeadV2_ResNet
from dataset import VOCDataset
from utils import (
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint
)
from loss import YoloLoss
from torch.utils.tensorboard import SummaryWriter

seed = 123
torch.manual_seed(seed)

# To update when running
# - update EXP_NAME
# - update DATA_DIR
# - update CLASSES (if different dataset)
# - update train/val files

# Hyperparameters etc.
EXP_NAME = "voc_aug_online_res34_v4"
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0.0005
EPOCHS = 150
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = f"checkpoints/{EXP_NAME}.pth.tar"
DATA_DIR = "data"
IMG_DIR = f"{DATA_DIR}/images"
CLASSES = 20
CELL_SPLIT = 7
IOU_THRESHOLD = 0.5
THRESHOLD = 0.4

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
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
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
    print(f"[Train] Mean Loss: {overall_mean_loss}")

    return overall_mean_loss

def val_loss(val_loader, model, loss_fn):
    loop = tqdm(val_loader, leave=True, disable=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        with torch.no_grad():
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

    overall_mean_loss = sum(mean_loss)/len(mean_loss)
    print(f"[Validation] Mean Loss: {overall_mean_loss}")

    return overall_mean_loss

def main():
    # model = Yolov1(split_size=CELL_SPLIT, num_boxes=2, num_classes=CLASSES).to(DEVICE)
    model = YOLOv1ResNet(backbone_name="resnet34", S=CELL_SPLIT, B=2, C=CLASSES).to(DEVICE)
    # model = YOLO_V1_HeadV2_ResNet(backbone_name="resnet34", S=CELL_SPLIT, B=2, C=CLASSES).to(DEVICE)
    loss_fn = YoloLoss(C=CLASSES, S=CELL_SPLIT)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        f"{DATA_DIR}/train.csv",  # 100examples.csv for quick checks
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=f"{DATA_DIR}/labels/",
        C=CLASSES,
        S=CELL_SPLIT,
        augment=True,
        aug_prob=0.6
    )

    val_dataset = VOCDataset(
        f"{DATA_DIR}/val.csv",
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=f"{DATA_DIR}/labels/",
        C=CLASSES,
        S=CELL_SPLIT
    )

    test_dataset = VOCDataset(
        f"{DATA_DIR}/test.csv",
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=f"{DATA_DIR}/labels/",
        C=CLASSES,
        S=CELL_SPLIT
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
    step = 0
    global_t = 0
    writer = SummaryWriter(f"logs/{EXP_NAME}")
    last_fc_layers = {
        "model.2.model.13.weight": "FC1",
        "model.2.model.16.weight": "FC2",
    }

    for epoch in range(EPOCHS):
        t_start = time.time()
        print(f"\n*** Epoch {epoch} ***")
        # train
        loss_train = train_fn(train_loader, model, optimizer, loss_fn)
        scheduler.step(loss_train)
        t_end = time.time()
        global_t += t_end - t_start
        loss_val = val_loss(val_loader, model, loss_fn)

        # add logs in tensorboard
        writer.add_scalars(
            f'Loss', 
            {
                'train': loss_train,
                'validation': loss_val,
            },
            global_step=step
        )
        # writer.add_scalar("Train Loss", loss_train, global_step=step)
        for name, param in model.named_parameters():
            if name in last_fc_layers.keys():
                writer.add_histogram(last_fc_layers[name], param, global_step=step)
        
        # perform mAP evaluation for every 5th epoch
        if epoch % 5 == 0 and epoch > 0:
            # get train pred results
            pred_boxes_train, target_boxes_train = get_bboxes(
                train_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD, num_classes=CLASSES, S=CELL_SPLIT
            )
            mean_avg_prec_train = mean_average_precision(
                pred_boxes_train, target_boxes_train, iou_threshold=IOU_THRESHOLD, box_format="midpoint", num_classes=CLASSES
            )
            print(f"[Train] mAP: {mean_avg_prec_train}")

            # get val pred results
            pred_boxes_val, target_boxes_val = get_bboxes(
                val_loader, model, iou_threshold=IOU_THRESHOLD, threshold=THRESHOLD, num_classes=CLASSES, S=CELL_SPLIT
            )
            mean_avg_prec_val = mean_average_precision(
                pred_boxes_val, target_boxes_val, iou_threshold=IOU_THRESHOLD, box_format="midpoint", num_classes=CLASSES
            )
            print(f"[Validation] mAP: {mean_avg_prec_val}")

            # add logs in tensorboard
            writer.add_scalars(
                f'mAP', 
                {
                    'train': mean_avg_prec_train,
                    'validation': mean_avg_prec_val,
                },
                global_step=step
            )

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

        step += 1
            
        # perform early stopping if no improvements for 20 epochs
        # if patience >= 4:
        #     print("\n***Early stopping executed at epoch {epoch} due to no improvements in validation mAP for the last 20 epochs***\n")
        #     print(f"\nTraining completed, total training time: {round(global_t / 3600, 2)} hrs\n")
        #     break

    print(f"\nTraining completed, total training time: {round(global_t / 3600, 2)} hrs\n")
                


if __name__ == "__main__":
    main()