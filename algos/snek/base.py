import math
import random

import torch
from torch import nn as nn

from abc import abstractmethod, ABC
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus


directions = [Richtung.Oben, Richtung.Rechts, Richtung.Unten, Richtung.Links]


def direction_to_action(direction: Richtung) -> int:
    if direction is None:
        return -1
    else:
        return directions.index(direction)


def action_to_direction(action: int) -> Richtung:
    if 0 <= action < len(directions):
        return directions[action]
    else:
        return None


field_states = [FeldZustand.Frei, FeldZustand.Wand, FeldZustand.Belegt, FeldZustand.Besucht]


def field_state_to_int(field_state: FeldZustand) -> int:
    if field_state is None:
        return -1
    else:
        return field_states.index(field_state)


def int_to_field_state(value: int) -> FeldZustand:
    if 0 <= value < len(field_states):
        return field_states[value]
    else:
        return None


class SnekBase(Algorithmus):

    epsilon_start = 0.9
    epsilon_end = 0.025
    epsilon_decay = 0.09

    min_exploration_steps = 3
    max_exploration_steps = 20

    def __init__(self, model: nn.Module):
        super().__init__(name="Snek")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._state: StateBase = None
        self._epsilon = self.epsilon_start

        self._max_turns = -1
        self._exploration_steps = -1

    @property
    def model(self) -> nn.Module:
        return self._model

    def _bereite_vor(self):
        self._max_turns = self.arena.punkte_maximum
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
        self._epsilon = self._calculate_epsilon(zug_nummer)

        if letzter_zustand.ist_blockiert:
            self._exploration_steps = min(self._exploration_steps, 0)
        if self._exploration_steps <= 0:
            if random.random() <= self._epsilon:
                except_directions = None if self._exploration_steps == -1 else [self.richtung, self.richtung.gegenteil]
                self._exploration_steps = self._start_exploration(zug_nummer)
                self._richtung = Richtung.zufall(ausser=except_directions)
            else:
                self._exploration_steps = -1

        if self._exploration_steps > 0:
            self._exploration_steps -= 1
            return self.richtung
        else:
            tensor = self._state.as_tensor(self._device)
            with torch.no_grad():
                prediction = self._model(tensor).flatten()
                maximum = prediction.max().item()
                indices = []
                for i, v in enumerate(prediction):
                    if v == maximum:
                        indices.append(i)
                action = random.choice(indices)
            return action_to_direction(action)

    def _calculate_epsilon(self, turn: int) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-(turn / self._max_turns) * (self.epsilon_decay * 100))

    def _start_exploration(self, turn: int) -> int:
        mode = self.max_exploration_steps - (self.max_exploration_steps - self.min_exploration_steps) * (turn / self._max_turns)
        mode = max(min(mode, self.max_exploration_steps), self.min_exploration_steps)
        steps = random.triangular(self.min_exploration_steps, self.max_exploration_steps, mode)
        return round(steps)

    @abstractmethod
    def _update_state(self, action_result: FeldZustand, turn: int, points: int):
        pass


class StateBase(ABC):

    @abstractmethod
    def as_tensor(self, device) -> torch.Tensor:
        pass

    @abstractmethod
    def copy(self):
        pass
