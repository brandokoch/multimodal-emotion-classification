from sys import exc_info
from typing import final
import numpy as np
from sklearn import metrics
import logging
import torch
import wandb

# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MoveToGPUCallback():
    def before_batch(self):
        try:
            self.learner.batch[0] = self.learner.batch[0].to('cuda')
            self.learner.batch[1] = self.learner.batch[1].to('cuda')
        except Exception as e:
            log.error(
                "Exception occurred: Can't move the batch to GPU", exc_info=True)

    def before_fit(self):
        try:
            self.learner.model = self.learner.model.to('cuda')
        except Exception as e:
            log.error(
                "Exception occurred: Can't move the model to GPU", exc_info=True)

class MultiModalMoveToGPUCallback():
    def before_batch(self):
        try:
            self.learner.batch[0][0] = self.learner.batch[0][0].to('cuda')
            self.learner.batch[0][1] = self.learner.batch[0][1].to('cuda')
            self.learner.batch[1] = self.learner.batch[1].to('cuda')
        except Exception as e:
            log.error(
                "Exception occurred: Can't move the batch to GPU", exc_info=True)

    def before_fit(self):
        try:
            self.learner.model = self.learner.model.to('cuda')
        except Exception as e:
            log.error(
                "Exception occurred: Can't move the model to GPU", exc_info=True)

class TrackResult():

    def before_epoch(self):
        self.batch_cnt = 0
        self.loss_sum = 0
        self.ys = []
        self.preds = []

    def after_batch(self):
        self.batch_cnt += 1
        loss = self.learner.loss
        _, yb = self.learner.batch
        preds = self.learner.preds

        yb = yb.detach().cpu().numpy().tolist()
        preds = preds.detach().cpu().numpy().tolist()
        loss = loss.detach().cpu()

        self.loss_sum += loss
        self.ys.extend(yb)
        self.preds.extend(preds)

        # Tracking train loss by batch
        if self.learner.model.training:
            lr= self.learner.sched.get_last_lr() #should be before batch WARNING
            wandb.log({'Loss/Train': loss, 'epoch': self.learner.epoch_idx, 'batch': self.learner.batch_idx})
            wandb.log({'Lr': lr[0], 'epoch': self.learner.epoch_idx, 'batch': self.learner.batch_idx})


    def after_epoch(self):

        # Calculate avg epoch loss
        avg_loss = self.loss_sum/self.batch_cnt

        # Calculate accuracy
        final_predictions = torch.tensor(self.preds)
        final_targets = torch.tensor(self.ys)

        final_predictions= torch.softmax(final_predictions, dim=1)
        final_predictions= torch.argmax(final_predictions, dim=1)

        final_predictions=torch.reshape(final_predictions, (-1,)).numpy()
        final_targets=torch.reshape(final_targets, (-1,)).numpy()

        accuracy = metrics.accuracy_score(final_targets, final_predictions)
        f1_score= metrics.f1_score(final_targets, final_predictions, average='weighted')

        # Log
        if self.learner.model.training:
            log.info(f"Epoch: {self.learner.epoch_idx} | Training | Loss: {avg_loss:.4f} | f1: {f1_score}")
            log.info(f"\n Confusion matrix: \n {metrics.confusion_matrix(final_targets, final_predictions)}")
            wandb.log({'Acc/Train': accuracy, 'epoch': self.learner.epoch_idx})
            wandb.log({'f1/Train': f1_score, 'epoch': self.learner.epoch_idx})
        else:

            log.info(f"Epoch: {self.learner.epoch_idx} | Validation | Loss: {avg_loss:.4f} | f1: {f1_score}")
            log.info(f"\n Confusion matrix: \n {metrics.confusion_matrix(final_targets, final_predictions)}")
            wandb.log({'Acc/Val': accuracy, 'epoch': self.learner.epoch_idx})
            wandb.log({'f1/Val': f1_score, 'epoch': self.learner.epoch_idx})



            # Tracking validation loss by epoch
            wandb.log({'Loss/Val': avg_loss, 'epoch': self.learner.epoch_idx})