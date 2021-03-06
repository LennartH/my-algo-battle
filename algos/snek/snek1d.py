import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Optional, Tuple
from algo_battle.domain import FeldZustand, Richtung
from snek.base import SnekBase, directions, StateBase, direction_to_action, field_state_to_int


class Snek1D(SnekBase):

    def __init__(self, model: "Snek1DModel", state_length=128, epsilon_decay: float = None):
        super().__init__(model, epsilon_decay)
        self._state_length = state_length
        self._previous_distances = (0, 0, 0, 0)

    def _bereite_vor(self):
        self._previous_distances = tuple(self.abstand(d) for d in directions)
        super()._bereite_vor()

    def _initialize_state(self) -> "Snek1DState":
        return Snek1DState([None for _ in range(self._state_length)])

    def _update_state(self, action_result: FeldZustand, turn: int, points: int):
        past_movements = self._state.past_movements[1:]
        latest_movement = Movement(self._previous_distances, self.richtung, action_result)
        past_movements.append(latest_movement)
        self._state.past_movements = past_movements
        self._previous_distances = tuple(self.abstand(d) for d in directions)


class Snek1DModel(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int, out_features: int):
        super().__init__()

        conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=2)
        conv2 = nn.Conv1d(in_channels=conv1.out_channels, out_channels=conv1.out_channels * 2, kernel_size=5, stride=2)

        conv3 = nn.Conv1d(in_channels=conv2.out_channels, out_channels=conv2.out_channels * 2, kernel_size=3)
        conv4 = nn.Conv1d(in_channels=conv3.out_channels, out_channels=64, kernel_size=3)

        self._features = nn.Sequential(
            conv1, nn.ReLU(inplace=True), nn.BatchNorm1d(conv1.out_channels),
            conv2, nn.ReLU(inplace=True), nn.BatchNorm1d(conv2.out_channels),
            nn.MaxPool1d(3),
            conv3, nn.ReLU(inplace=True), nn.BatchNorm1d(conv3.out_channels),
            conv4, nn.ReLU(inplace=True), nn.BatchNorm1d(conv4.out_channels),
            nn.AvgPool1d(conv4.out_channels),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(conv4.out_channels, out_features=out_features)

    def forward(self, x: torch.Tensor):
        x = self._features(x)
        x = self.head(x.view(x.size(0), -1))
        return F.softmax(x, dim=0)


@dataclass
class Snek1DState(StateBase):

    past_movements: List[Optional["Movement"]]

    def as_tensor(self, device) -> torch.Tensor:
        data = self._as_array()
        tensor = torch.from_numpy(data)
        return tensor.transpose(0, 1).unsqueeze(0).float().to(device)

    def _as_array(self) -> np.ndarray:
        packed = []
        for movement in self.past_movements:
            if not movement:
                movement_array = [-1 for _ in range(Movement.size())]
            else:
                movement_array = [
                    *movement.distances,
                    direction_to_action(movement.direction) + 1,
                    field_state_to_int(movement.result) + 1
                ]
            packed.append(movement_array)
        return np.asarray(packed)

    def copy(self):
        return Snek1DState(list(self.past_movements))


@dataclass
class Movement:

    distances: Tuple[int, int, int, int]
    direction: Richtung
    result: FeldZustand

    @staticmethod
    def size():
        return 6
