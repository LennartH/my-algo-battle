import math
import random

import numpy as np
import torch
from torch import nn as nn

from abc import abstractmethod, ABC
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus


directions = [Richtung.Oben, Richtung.Rechts, Richtung.Unten, Richtung.Links]
field_states = [FeldZustand.Frei, FeldZustand.Wand, FeldZustand.Belegt, FeldZustand.Besucht]


class SnekBase(Algorithmus):

    epsilon_start = 0.9
    epsilon_end = 0.025
    epsilon_decay = 0.0005

    def __init__(self, model: nn.Module):
        super().__init__(name="Snek")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._state: StateBase = None
        self._epsilon = 0.0
        self._update_epsilon(0)

    @property
    def model(self) -> nn.Module:
        return self._model

    def _bereite_vor(self):
        self._state = self._initialize_state()
        # Perform initial prediction to load torch context
        # This would cost around 400 turns if done during the competition
        with torch.no_grad():
            self._model(self._state.as_tensor(self._device))

    @abstractmethod
    def _initialize_state(self) -> "StateBase":
        pass

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        self._update_state(letzter_zustand, zug_nummer, aktuelle_punkte)
        tensor = self._state.as_tensor(self._device)
        with torch.no_grad():
            prediction = self._model(tensor).flatten()
            maximum = prediction.max().item()
            indices = []
            for i, v in enumerate(prediction):
                if v == maximum:
                    indices.append(i)
            index = random.choice(indices)
        return directions[index]

    def _update_epsilon(self, turn: int):
        self._epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-turn * self.epsilon_decay)

    @abstractmethod
    def _update_state(self, action_result: FeldZustand, turn: int, points: int):
        pass


class StateBase(ABC):

    def as_tensor(self, device) -> torch.Tensor:
        data = self._as_array()
        tensor: torch.Tensor = torch.from_numpy(data)
        return tensor.to(device)

    @abstractmethod
    def _as_array(self) -> np.ndarray:
        pass

    @abstractmethod
    def copy(self):
        pass
