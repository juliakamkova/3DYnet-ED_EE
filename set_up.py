import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import utils
import monai
# from monai.data import CacheDataset
from monai.data import CacheDataset, list_data_collate, decollate_batch
from monai.metrics import compute_meandice, DiceMetric
from monai.inferers import sliding_window_inference
# import glob
from pathlib import Path
import evaluation
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor
from box import Box


# TODO: IMPLEMENTS MORE METRICS DURING THE TRAINING AND VALIDATION

class Framework(pl.LightningModule):
    def __init__(self, net, params, n_cl):
        super().__init__()
        self.net = net
        self.params = params
        self.n_cl = n_cl
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=n_cl, n_classes=n_cl)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=n_cl, n_classes=n_cl)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False,
                                      ignore_empty=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        # self.check_val = 30 #!TODO wtf is this
        # self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, x):
        return self.net(x)  # F.softmax(self.net(x), dim=1)

    def prepare_data(self):
        transforms = []
        for class_params in self.params.train.augmentation:
            # print('checking which train augmentation is loading ', class_params)
            transforms.extend(utils.class_loader(class_params))
        train_transform = monai.transforms.Compose(transforms)

        transforms = []
        for class_params in self.params.valid.augmentation:
            # print('checking which train augmentation in validation is loading ', class_params)
            transforms.extend(utils.class_loader(class_params))
        valid_transform = monai.transforms.Compose(transforms)

        train_images = sorted(Path(self.params.data.image).glob('train/*.nii*'))
        train_labels = sorted(Path(self.params.data.label).glob('train/*.nii*'))
        # print('training dataset images' ,train_images)

        train_dicts = [
            {"image": str(image_name), "label": str(label_name)}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        valid_images = sorted(Path(self.params.data.image).glob('valid/*.nii*'))
        valid_labels = sorted(Path(self.params.data.label).glob('valid/*.nii*'))
        # print('validation dataset images' ,valid_images)
        # print('validation dataset labels', valid_labels)

        valid_dicts = [
            {"image": str(image_name), "label": str(label_name)}
            for image_name, label_name in zip(valid_images, valid_labels)
        ]

        self.training_ds = CacheDataset(data=train_dicts, transform=train_transform, cache_num=24, cache_rate=1.0,
                                        num_workers=12)  # num_workers=8)

        self.validation_ds = CacheDataset(data=valid_dicts, transform=valid_transform, cache_num=6, cache_rate=1.0,
                                          num_workers=12)
        # self.test_ds = CacheDataset(data=valid_dicts, transform=valid_transform, num_workers=4)

        self.loss_function = utils.class_loader(self.params.loss)[0]
        print('from prepare data note: loss function that is used: ', self.loss_function)

    def train_dataloader(self):
        self.training_dataloader = torch.utils.data.DataLoader(self.training_ds,
                                                               collate_fn=list_data_collate,
                                                               batch_size=1,
                                                               shuffle=True,
                                                               pin_memory=True)
        return self.training_dataloader

    def val_dataloader(self):
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_ds, batch_size=1, shuffle=False,
                                                                 pin_memory=True)
        return self.validation_dataloader

    # def test_dataloader(self):
    #     self.test_dataloader = torch.utils.data.DataLoader(self.test_ds,**self.params['valid']['loader'])
    #     return self.test_dataloader

    def configure_optimizers(self):
        optim = utils.class_loader(self.params.optimizer, extra_args=(self.net.parameters(),))[0]
        print('from config optimizer check optim', optim)
        return [optim]

    def training_step(self, batch, batch_idx):
        input = batch['image'].cuda()  # shape[batch, channels, dim, dim , dim]  !TODO to device???
        mask = batch['label'].cuda()
        output = self.forward(input)

        # evaluation.save_batch(input.argmax(dim=0).float(), 'check_volume_15_09.nii')
        # evaluation.save_batch(mask.argmax(dim=0).float(), 'check_mask_15_09.nii')

        train_loss = self.loss_function(output, mask)
        # print('train loss inside the lightning', train_loss)

        output1 = [self.post_pred(i) for i in decollate_batch(output)]
        mask1 = [self.post_label(i) for i in decollate_batch(mask)]

        train_monai_dice = self.dice_metric(y_pred=output1, y=mask1)

        return {"loss": train_loss, "train_step_dice": train_monai_dice}

    def training_epoch_end(self, outputs):
        train_dice_per_class = torch.zeros(2).cuda() # Initialize per-class dice scores
        train_loss1, num_items1 = 0, 0

        for output in outputs:
            train_dice_per_class += output["train_step_dice"].sum(
                dim=0).cuda() # Sum along rows to accumulate class-wise dice scores
            train_loss1 += output["loss"].sum().item()
            num_items1 += len(output["train_step_dice"])

        mean_train_dice_per_class = train_dice_per_class / num_items1  # Calculate mean per-class dice scores
        mean_train_dice = mean_train_dice_per_class.mean()  # Calculate the overall mean dice score

        mean_train_loss = train_loss1 / num_items1

        self.log('train_epoch_Dice', mean_train_dice)
        self.log('train_epoch_loss', mean_train_loss)

    def validation_step(self, batch, batch_idx):
        input = batch['image']
        masks = batch['label']
        outputs = self.forward(input)
        # print('outputs before val loss and inside the lightning', outputs)
        # print('mask', masks)

        val_loss = self.loss_function(outputs, masks)

        # evaluation.save_batch(input.argmax(dim=0).float(), 'valid_check_volume_15_09.nii')
        # evaluation.save_batch(masks.argmax(dim=0).float(), 'valid_check_mask_15_09.nii')
        outputs2 = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels2 = [self.post_label(i) for i in decollate_batch(masks)]
        monai_dice = self.dice_metric(y_pred=outputs2, y=labels2)
        print('from validation step checking Dice', monai_dice)
        print('from validation step checking Loss', val_loss)

        self.log('val step dice', monai_dice.mean())

        return {"val_loss": val_loss, "val_step_dice": monai_dice}

    def validation_epoch_end(self, outputs):
        val_dice_per_class = torch.zeros(2).cuda()  # Initialize per-class dice scores
        val_loss, num_items = 0, 0

        for output in outputs:
            val_dice_per_class += output["val_step_dice"].sum(
                dim=0).cuda()  # Sum along rows to accumulate class-wise dice scores
            val_loss += output["val_loss"].sum().item()
            num_items += len(output["val_step_dice"])

        mean_val_dice_per_class = val_dice_per_class / num_items  # Calculate mean per-class dice scores
        mean_val_dice = mean_val_dice_per_class.mean()  # Calculate the overall mean dice score

        mean_val_loss = val_loss / num_items
        print('from validation epoch end checking overall mean validation dice', mean_val_dice)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        self.log('val_epoch_Dice', mean_val_dice)
        self.log('val_loss', mean_val_loss)
        self.log('Best validation Dice', self.best_val_dice)

        print(
            f"\ncurrent epoch: {self.current_epoch} current val mean dice: {mean_val_dice:.4f} current val loss {mean_val_loss:.3f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} at epoch: {self.best_val_epoch:.4f}"
        )
