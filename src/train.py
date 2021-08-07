import config

from logging import error, log
from sys import exc_info
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.logconf import logging
import wandb
from datasets import get_dataloaders
from learner import Learner
import model_dispatcher
import callback_dispatcher

# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class TrainingApp:
    def __init__(self):

        log.info('----- Training Started -----')

        # Device handling
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        log.info(f'GPU availability: {self.use_cuda}')
        log.info(f'Device name is {torch.cuda.get_device_name()}')

    def main(self):
        train_dl, val_dl = get_dataloaders(wandb.config.DATASET)

        try:
            model = model_dispatcher.models[wandb.config.MODEL]
            wandb.watch(model, log_freq=100)
            loss_func = getattr(nn, wandb.config.LOSS)(weight=None)#wandb.config.CLASS_WEIGHTS.to(self.device))
            opt_func = getattr(optim, wandb.config.OPTIMIZER)
            scheduler_func=getattr(optim.lr_scheduler, wandb.config.SCHEDULER)
            cbs = callback_dispatcher.callbacks[wandb.config.CBS]
        except Exception as e:
            log.error(
                "Exception occurred: Configuration is invalid, check the README", exc_info=True)

        learner = Learner(model, train_dl, val_dl, loss_func,
                          wandb.config.LR, wandb.config.WEIGHT_DECAY, cbs, opt_func, scheduler_func)
        learner.fit(wandb.config.EPOCHS)
        learner.save(os.path.join(wandb.config.RUNS_FOLDER_PTH,wandb.config.RUN_NAME, wandb.config.MODEL+'.pt'))


if __name__ == "__main__":
    TrainingApp().main()