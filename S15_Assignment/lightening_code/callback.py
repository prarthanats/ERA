
from config import get_config, get_weights_file_path 
import pytorch_lightning as pl
import torch 


class TrainEndCallback(pl.Callback):
    def __init__(self, config):
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        average_loss = sum(pl_module.training_loss) / len(pl_module.training_loss)
        
        print(f'Epoch Number {epoch}')
        print(f'Average Loss {average_loss}')

        model_filename = get_weights_file_path(self.config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': pl_module.model.state_dict(),
            'optimizer_state_dict': pl_module.optimizer.state_dict(),
            'global_step': trainer.global_step
        }, model_filename)
