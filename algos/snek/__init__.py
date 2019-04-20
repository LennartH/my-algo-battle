import os
import numpy as np
import torch
import torch.nn as nn
import random

from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus
from snek.model import Snek1DModel

directions = [Richtung.Oben, Richtung.Rechts, Richtung.Unten, Richtung.Links]
field_states = [FeldZustand.Frei, FeldZustand.Wand, FeldZustand.Belegt, FeldZustand.Besucht]


class SnekBase(Algorithmus):

    def __init__(self, model: nn.Module):
        super().__init__(name="Snek")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._state: StateBase = None

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

    @abstractmethod
    def _update_state(self, action_result: FeldZustand, turn: int, points: int):
        pass


class Snek1D(SnekBase):

    def __init__(self, model_state_path: str = None, state_length=250):
        m = Snek1DModel(in_channels=Movement.size(), kernel_size=10, out_features=len(directions))
        if model_state_path and os.path.isfile(model_state_path):
            m.load_state_dict(torch.load(model_state_path))
        m.eval()
        super().__init__(m)
        self._state_length = state_length
        self._previous_distances = (0, 0, 0, 0)

    def _bereite_vor(self):
        self._previous_distances = tuple(self.abstand(d) for d in directions)
        super()._bereite_vor()

    def _initialize_state(self) -> "Snek1DState":
        return Snek1DState([None for _ in range(self._state_length)])

    def _update_state(self, action_result: FeldZustand, turn: int, points: int):
        past_movements = self._state.past_movements[:-1]
        latest_movement = Movement(self._previous_distances, self.richtung, action_result)
        past_movements.append(latest_movement)
        self._state.past_movements = past_movements
        self._previous_distances = tuple(self.abstand(d) for d in directions)


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


@dataclass
class Snek1DState(StateBase):

    past_movements: List[Optional["Movement"]]

    def _as_array(self) -> np.ndarray:
        packed = []
        for movement in self.past_movements:
            if not movement:
                movement_array = [0 for _ in range(Movement.size())]
            else:
                movement_array = [
                    *movement.distances,
                    directions.index(movement.direction) + 1,
                    field_states.index(movement.result) + 1
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
