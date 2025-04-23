import os
from typing import Optional

import torch
import omegaconf
from loguru import logger

from training.trainer import Trainer
from models import get_model
from datasets.guidedepth_dataset import GuideDepthDataset
from training.metrics import plot_metrics

from PEERNet_fl.peernet.networks import ZMQ_Pair

# Initialize Model
model_name = "hybrid"
model_params = {"in_channels":3, "height":240, "width":320}
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_model(model_name, **model_params).to(device)

training_data_paths = ["train_data/nyu_data/data/nyu2_train/dining_room_0010_out", "train_data/nyu_data/data/nyu2_train/basement_0001b_out", "train_data/nyu_data/data/nyu2_train/classroom_0004_out", "train_data/nyu_data/data/nyu2_train/basement_0001a_out"]
# training_data_paths = ["/home/student/edgemapper-client/train_data/nyu_data/data/nyu2_train/bedroom_0060_out", "/home/student/edgemapper-client/train_data/nyu_data/data/nyu2_train/bedroom_0060_out", "/home/student/edgemapper-client/train_data/nyu_data/data/nyu2_train/bedroom_0060_out", "/home/student/edgemapper-client/train_data/nyu_data/data/nyu2_train/bedroom_0060_out"]
val_data_path = "/home/student/edgemapper-client/train_data/nyu_data/data/nyu2_train/basement_0001a_out"

trainer = Trainer(
    model,
    model_name,
    training_data_paths[0],
    device,
    4,
    1e-4,
    val_data_path
)
logger.info(f"Initialized trainer on {device}")

global_epoch = 0
local_epoch = 0
max_local_epochs = 3


global_metrics = {}
for idx, training_path in enumerate(training_data_paths):
    trainer.plot_val()
    trainer.train(max_local_epochs)
    local_metrics = trainer.validate()
    # global_metrics.update({idx, local_metrics})
    trainer.update_dataset(training_path)
trainer.plot_val()
trainer.plot_results()
trainer.
# plot_metrics(global_metrics, "./results")
