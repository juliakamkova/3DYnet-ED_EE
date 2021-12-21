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
# import networks
from box import Box

class Framework(pl.LightningModule):
    def __init__(self, net, params, n_cl):
        super().__init__()
        self.net = net
        self.params = params
        self.n_cl = n_cl
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=n_cl)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=n_cl)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.check_val = 3
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []


    def forward(self, x):
        return self.net(x) #F.softmax(self.net(x), dim=1)#
        

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
        self.training_ds = CacheDataset(data=train_dicts, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=4)# num_workers=8)

        self.validation_ds = CacheDataset(data=valid_dicts, transform=valid_transform, cache_num=6, cache_rate=1.0, num_workers=4)
        self.test_ds = CacheDataset(data=valid_dicts, transform=valid_transform, num_workers=4)
        self.loss_function = utils.class_loader(self.params.loss)[0]

        print('from prepare data note: loss function that is used: ', self.loss_function)

    def train_dataloader(self):
        self.training_dataloader = torch.utils.data.DataLoader(self.training_ds,
                                                               collate_fn=list_data_collate,
                                                               batch_size=1,
                                                               shuffle=True,
                                                               pin_memory=True,)
        return self.training_dataloader

    def val_dataloader(self):
        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_ds, batch_size=1, shuffle=False, pin_memory=True)
        return self.validation_dataloader

    # def test_dataloader(self):
    #     self.test_dataloader = torch.utils.data.DataLoader(self.test_ds,**self.params['valid']['loader'])
    #     return self.test_dataloader

    def configure_optimizers(self):
        #name_optimizer = list(self.params.optimizer.keys())[0]
        #self.lr = self.params.optimizer[name_optimizer].kwargs.lr
        optim = utils.class_loader(self.params.optimizer, extra_args=(self.net.parameters(),))[0]
        print('from config optimizer check optim',optim )
        # lr_scheduler = {
        #     #"scheduler": utils.class_loader(self.params.scheduler, extra_args=self.params.optimizer(), self.net.parameters(), )[0], !TODO
        #     #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode = 'min', cooldown = 5),
        #     "monitor": 'val_loss',
        #     'name': 'LR_logging_name',
        #     'interval': 'epoch',
        #     'frequency': 1
        #     }
        #print('from cofigure otimizer note: optimizer that is used: ', optim, 'check learning rate', self.lr, 'lr scedulare', lr_dict["scheduler"] )
        return [optim]#, [lr_scheduler]

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

        #train_monai_dice = self.dice_metric(y_pred=output1, y=mask1)

        summarywriter = self.logger.experiment
        if batch_idx == 0:
            evaluation.generate_gallery(input[0,...],
                                        mask[0,...],
                                        output[0,...],
                                        summarywriter,
                                        'train',
                                        global_step=self.global_step)

        return {"loss": train_loss} #, "train_step_dice": train_monai_dice}

    def training_epoch_end(self, outputs):
        # print('learning rate', self.lr)
        # self.log('learning_rate_training', self.lr)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
        #
        # train_dice1, train_loss1, num_items1 = 0, 0, 0
        # for output in outputs:
        #     train_dice1 += output["train_step_dice"].sum().item()
        #     train_loss1 += output["loss"].sum().item()
        #     num_items1 += len(output["train_step_dice"])
        #
        #
        #
        # mean_train_dice = torch.tensor(train_dice1 / num_items1)
        # mean_train_loss = torch.tensor(train_loss1 / num_items1)
        #
        # print('from training epoch end DICE', mean_train_dice)
        #
        # self.log('train_epoch_Dice', mean_train_dice)
        # self.log('train_epoch_loss', mean_train_loss)
       # return {"val_loss": 1}

    def validation_step(self, batch, batch_idx):
        input = batch['image']
        masks = batch['label']

        print(self.params.data.patch)
        roi_size = list(self.params.data.patch) #!TODO
        sw_batch_size = 1
        outputs = sliding_window_inference(input, roi_size, sw_batch_size, self.forward)
        val_loss = self.loss_function(outputs, masks)

        #print('val loss inside the lightning', val_loss)
        self.log('val_loss', val_loss)
        #print('check if this add logs.keys', list(logs.keys()))
        #outputs = self.forward(input)
        # evaluation.save_batch(input.argmax(dim=0).float(), 'valid_check_volume_15_09.nii')
        # evaluation.save_batch(masks.argmax(dim=0).float(), 'valid_check_mask_15_09.nii')

        #print('from validation step', outputs.shape, masks.shape)

        outputs2 = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels2 = [self.post_label(i) for i in decollate_batch(masks)]

        # print('from validation second', outputs2.shape, labels2.shape)
        self.dice_metric(y_pred=outputs2, y=labels2)

        monai_dice = self.dice_metric(y_pred=outputs2,y=labels2)
        # hausdoff =
        # own_metric = (monai_dice+hausdoff)/2
        print('from validation step checking Dice', monai_dice, batch_idx)
        self.log('val step Dice', monai_dice)


        summarywriter = self.logger.experiment
        if batch_idx == 0:
            evaluation.generate_gallery(input[0,...],
                                        masks[0,...],
                                        outputs[0,...],
                                        summarywriter,
                                        'valid',
                                        global_step=self.global_step)

        return {"val_loss": val_loss, "val_number": len(outputs)}  #"val_step_dice": monai_dice}


    def validation_epoch_end(self, outputs):
        val_dice, val_loss, num_items = 0, 0, 0
        for output in outputs:
            # val_dice += output["val_step_dice"].sum().item()
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item() #i think this belongs to patch version of the code
        self.dice_metric.reset()

        #mean_val_dice = torch.tensor(val_dice / num_items)
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        self.log('val_epoch_Dice', mean_val_dice)
        self.log('val_epoch_Loss', mean_val_loss)
        self.log('Best validation Dice', self.best_val_dice)

        print(
            f"\ncurrent epoch: {self.current_epoch}"
            f"\ncurrent val mean dice: {mean_val_dice:.4f} "
            f"\ncurrent val loss {mean_val_loss:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch:.4f}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}

    # def test_step(self, batch, batch_idx):
    #     print('we are in the beggining of the test step')
    #     input = batch['image']
    #     mask = batch['label']
    #     output = self.forward(input)
    #     loss = self.loss_function(output, mask)
    #     self.log('test_loss', loss)
    #     #output1 = F.one_hot(output, num_classes = 3)
    #     monai_dice = self.dice_metric(y_pred=output1,
    #                     y=mask)
    #     monai_HD95 = monai.metrics.hausdorff_distance(y_pred=output1,y=mask)
    #
    #     prediction = output[0,...]
    #     prediction = prediction.detach().cpu().argmax(dim=0).float()
    #     #evaluation.save_batch(prediction, 'test_try.nii')
    #     print('test_loss', loss, 'test_dice', monai_dice, 'test_hausdoff', monai_HD95)
    #     return {"test_loss": loss, "test_dice": monai_dice}
    #
    #
    # def test_end(self, outputs):
    #    avg_loss= torch.stack([x['test_loss'] for x in outputs]).mean()
    #    tensorboard_logs = {'xxavg_loss_for_test': avg_loss}
    #    return {'xtest_loss': avg_loss, 'log': tensorboard_logs}
