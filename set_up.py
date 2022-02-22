#import numpy as np
#import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import utils 
import monai
#from monai.data import CacheDataset
from monai.data import CacheDataset,list_data_collate, decollate_batch
from monai.metrics import compute_meandice , DiceMetric
from monai.inferers import sliding_window_inference
#import glob
from pathlib import Path
import evaluation
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor
from box import Box

class Framework(pl.LightningModule):
    def __init__(self, net, params, n_cl):
        super().__init__()
        self.net = net
        self.params = params
        self.n_cl = n_cl
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=n_cl)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=n_cl)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        # self.check_val = 30 !TODO wtf is this
        # self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []


    def forward(self, x):
        return self.net(x)#F.softmax(self.net(x), dim=1)
        

    def prepare_data(self):
        transforms = []
        for class_params in self.params.train.augmentation:
            #print('checking which train augmentation is loading ', class_params)
            transforms.extend(utils.class_loader(class_params))
        train_transform = monai.transforms.Compose(transforms)
        
        transforms = []
        for class_params in self.params.valid.augmentation:
            #print('checking which train augmentation in validation is loading ', class_params)
            transforms.extend(utils.class_loader(class_params))   
        valid_transform = monai.transforms.Compose(transforms)

        train_images = sorted(Path(self.params.data.image).glob('train/*.nii*'))
        train_labels = sorted(Path(self.params.data.label).glob('train/*.nii*'))
        #print('training dataset images' ,train_images)

        train_dicts = [
            {"image": str(image_name), "label": str(label_name)}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        valid_images = sorted(Path(self.params.data.image).glob('valid/*.nii*'))
        valid_labels = sorted(Path(self.params.data.label).glob('valid/*.nii*'))
        #print('validation dataset images' ,valid_images)
        #print('validation dataset labels', valid_labels)

        valid_dicts = [
            {"image": str(image_name), "label": str(label_name)}
            for image_name, label_name in zip(valid_images, valid_labels)
        ]
        
        # test_images = sorted(Path(self.params.data.image).glob('test/*.nii*'))
        # test_labels = sorted(Path(self.params.data.label).glob('test/*.nii*'))

        # test_dicts = [
        #     {"image": str(image_name), "label": str(label_name)}
        #     for image_name, label_name in zip(test_images, test_labels)
        # ]
        

        #set deterministic training for reproducibility
        #monai.utils.set_determinism(seed=0)
        #print('\n')
        self.training_ds = CacheDataset(data=train_dicts, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=12)# num_workers=8)

        self.validation_ds = CacheDataset(data=valid_dicts, transform=valid_transform, cache_num=6, cache_rate=1.0, num_workers=12)
        #self.test_ds = CacheDataset(data=valid_dicts, transform=valid_transform, num_workers=4)

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
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_ds, batch_size=1, shuffle=False, pin_memory=True)
        return self.validation_dataloader

    # def test_dataloader(self):
    #     self.test_dataloader = torch.utils.data.DataLoader(self.test_ds,**self.params['valid']['loader'])
    #     return self.test_dataloader

    def configure_optimizers(self):
        optim = utils.class_loader(self.params.optimizer, extra_args=(self.net.parameters(),))[0]
        print('from config optimizer check optim', optim)
        return [optim]

    def training_step(self, batch, batch_idx):
        input = batch['image'].cuda()  #shape[batch, channels, dim, dim , dim]  !TODO to device???
        mask = batch['label'].cuda()
        output = self.forward(input)

        # evaluation.save_batch(input.argmax(dim=0).float(), 'check_volume_15_09.nii')
        # evaluation.save_batch(mask.argmax(dim=0).float(), 'check_mask_15_09.nii')

        train_loss = self.loss_function(output, mask)
        #print('train loss inside the lightning', train_loss)

        output1 = [self.post_pred(i) for i in decollate_batch(output)]
        mask1 = [self.post_label(i) for i in decollate_batch(mask)]

        train_monai_dice = self.dice_metric(y_pred=output1, y=mask1)

        # summarywriter = self.logger.experiment
        # if batch_idx == 0:
        #     evaluation.generate_gallery(input[0,...],
        #                                 mask[0,...],
        #                                 output[0,...],
        #                                 summarywriter,
        #                                 'train',
        #                                 global_step=self.global_step)

        return {"loss": train_loss, "train_step_dice": train_monai_dice}

    def training_epoch_end(self, outputs):
        # print('learning rate', self.lr)
        # self.log('learning_rate_training', self.lr)

        train_dice1, train_loss1, num_items1 = 0, 0, 0
        for output in outputs:
            train_dice1 += output["train_step_dice"].sum().item()
            train_loss1 += output["loss"].sum().item()
            num_items1 += len(output["train_step_dice"])



        mean_train_dice = torch.tensor(train_dice1 / num_items1)
        mean_train_loss = torch.tensor(train_loss1 / num_items1)

        print('from training epoch end DICE', mean_train_dice)

        self.log('train_epoch_Dice', mean_train_dice)
        self.log('train_epoch_loss', mean_train_loss)

    def validation_step(self, batch, batch_idx):
        input = batch['image']
        masks = batch['label']

        # roi_size = (96, 96, 96) #!TODO
        # sw_batch_size = 1
        # outputs = sliding_window_inference(
        # input, roi_size, sw_batch_size, self.forward)
        outputs = self.forward(input)

        val_loss = self.loss_function(outputs, masks)
        #print('val loss inside the lightning', val_loss)
        # self.log('Val step loss', val_loss)

        # evaluation.save_batch(input.argmax(dim=0).float(), 'valid_check_volume_15_09.nii')
        # evaluation.save_batch(masks.argmax(dim=0).float(), 'valid_check_mask_15_09.nii')

        # print('from validation step', mask.shape, output.shape)

        outputs2 = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels2 = [self.post_label(i) for i in decollate_batch(masks)]
        #self.dice_metric(y_pred=outputs, y=labels)

        monai_dice = self.dice_metric(y_pred=outputs2,y=labels2)
        print('from validation step checking Dice', monai_dice, batch_idx)

        # print('val_step_Dice_liver', monai_dice[0])
        # print('val_step_Dice_liver', monai_dice[1])
        # self.log('val step Dice', monai_dice)

        # summarywriter = self.logger.experiment
        # if batch_idx == 0:
        #     evaluation.generate_gallery(input[0,...],
        #                                 masks[0,...],
        #
        #                                 outputs[0,...],
        #                                 summarywriter,
        #                                 'valid',
        #                                 global_step=self.global_step)

        # self.log('val_step_Dice_liver', monai_dice[0])
        # self.log('val_step_Dice_tumor', monai_dice[1])

        return {"val_loss": val_loss, "val_step_dice": monai_dice}


    def validation_epoch_end(self, outputs):
        val_dice, val_loss, num_items = 0, 0, 0
        for output in outputs:
            val_dice += output["val_step_dice"].sum().item()
            val_loss += output["val_loss"].sum().item()
            num_items += len(output["val_step_dice"])

        #mean_val_dice = self.dice_metric.aggregate().item() #i think this belongs to patch version of the code
        #self.dice_metric.reset()

        mean_val_dice = torch.tensor(val_dice / num_items)
        mean_val_loss = torch.tensor(val_loss / num_items)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        self.log('val_epoch_Dice', mean_val_dice)
        self.log('val_loss', mean_val_loss)
        self.log('Best validation Dice', self.best_val_dice)

        print(
            f"current epoch: {self.current_epoch} current val mean dice: {mean_val_dice:.4f} current val loss {val_loss:.3f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} at epoch: {self.best_val_epoch:.4f}"
        )


    # def test_step(self, batch, batch_idx):
    #     print('we are in the beggining of the test step')
    #     input = batch['image']
    #     mask = batch['label']
    #     output = self.forward(input)
    #     loss = self.loss_function(output, mask)
    #     self.log('test_loss', loss)
    #     output1 = F.one_hot(output, num_classes = 3)
    #     monai_dice = self.dice_metric(y_pred=output1,  #!TODO
    #                     y=mask)
    #
    #     prediction = output[0,...]
    #     prediction = prediction.detach().cpu().argmax(dim=0).float()
    #     #evaluation.save_batch(prediction, 'test_try.nii')
    #     print('test_loss', loss, 'test_dice', monai_dice)
    #     return {"test_loss": loss, "test_dice": monai_dice}


# def test_end(self, outputs):
#        avg_loss= torch.stack([x['test_loss'] for x in outputs]).mean()
#        tensorboard_logs = {'xxavg_loss_for_test': avg_loss}
#        return {'xtest_loss': avg_loss, 'log': tensorboard_logs}
