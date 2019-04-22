import os
import torch

from .base import directions
from .snek1d import Snek1D, Snek1DModel, Movement


class Snek(Snek1D):

    def __init__(self):
        model_state_path = "snek1d_state.pth"
        model = Snek1DModel(in_channels=Movement.size(), kernel_size=10, out_features=len(directions))
        if os.path.isfile(model_state_path):
            print("Loaded Model")
            model.load_state_dict(torch.load(model_state_path))
        model.eval()
        super().__init__(model)
