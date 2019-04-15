import os
import numpy as np
import tensorflow as tf
import random

from typing import List, Tuple, Iterable
from dataclasses import dataclass, astuple
from keras import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus


directions = [Richtung.Oben, Richtung.Rechts, Richtung.Unten, Richtung.Links]
field_states = [FeldZustand.Frei, FeldZustand.Wand, FeldZustand.Belegt, FeldZustand.Besucht]

rewards_map = {
    FeldZustand.Frei: 1,
    FeldZustand.Wand: -1,
    FeldZustand.Belegt: -0.5,
    FeldZustand.Besucht: -0.1
}

number_of_past_actions_in_state = 100
short_term_memory_size = 10


class Snek(Algorithmus):

    def __init__(self, weights_path="weights.hdf5", learning_rate=0.0005, gamma=0.9, train_while_running=True):
        super().__init__()
        self._gamma = gamma
        self._model = self._create_model(learning_rate, weights_path)
        self._graph = tf.get_default_graph()
        self._memory: List[MemoryEntry] = []
        self._train_while_running = train_while_running
        self._turns_till_training = short_term_memory_size
        self._q_table = {d: random.random() for d in directions}

    @staticmethod
    def _create_model(learning_rate: float, weights_path: str) -> Model:
        input_length = State.tensor_length(number_of_past_actions_in_state)
        output_length = len(directions)
        hidden_units = (input_length + output_length) // 2

        model = Sequential()
        model.add(Dense(hidden_units, activation="relu", input_dim=input_length))
        model.add(Dropout(rate=0.15))
        model.add(Dense(hidden_units, activation="relu"))
        model.add(Dropout(rate=0.15))
        model.add(Dense(hidden_units, activation="relu"))
        model.add(Dropout(rate=0.15))
        model.add(Dense(units=output_length, activation="softmax"))
        model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        return model

    # TODO Move to framework
    @property
    def position(self) -> Tuple[int, int]:
        return self._x, self._y

    @property
    def model(self) -> Model:
        return self._model

    def _bereite_vor(self):
        # noinspection PyTypeChecker
        empty_actions = [Action(None, None) for _ in range(number_of_past_actions_in_state)]
        initial_state = State(self.position, self.arena.form, empty_actions, turn=0, points=0)
        self._memory.append(MemoryEntry(initial_state, 0))

    def _gib_richtung(self, result: FeldZustand, turn: int, points: int) -> Richtung:
        previous_state = self._memory[-1].state
        actions = list(previous_state.past_actions)[:-1]
        actions.append(Action(self.richtung, result))
        state = State(self.position, self.arena.form, actions, turn, points)
        reward = rewards_map[result]
        self._memory.append(MemoryEntry(state, reward))
        if self._train_while_running:
            if self._turns_till_training <= 0:
                self._train(short_term_memory_size)
                self._turns_till_training = short_term_memory_size
            self._turns_till_training -= 1
        return self._predict_direction(state)

    def _predict_direction(self, state: "State") -> Richtung:
        with self._graph.as_default():
            input = np.expand_dims(state.as_tensor(), axis=0)
            prediction = self._model.predict(input)
            # noinspection PyTypeChecker
            direction_index: int = np.argmax(prediction)
            return directions[direction_index]

    def _train(self, n: int = None):
        memory_slice = self._memory[len(self._memory) - n:] if n and n < len(self._memory) else self._memory
        epochs = len(memory_slice)
        # TODO Fresh q-table per training? Knowledge in NN?
        for i, memory in enumerate(memory_slice):
            input = np.expand_dims(memory.state.as_tensor(), axis=0)
            # FIXME Correct Bellman equation
            self._q_table[memory.direction] += self._gamma * memory.reward
            target = np.expand_dims(np.asarray([self._q_table[d] for d in directions]), axis=0)
            with self._graph.as_default():
                self._model.fit(x=input, y=target, epochs=1, verbose=0)



@dataclass
class MemoryEntry:

    state: "State"
    reward: float

    @property
    def direction(self) -> Richtung:
        return self.state.past_actions[-1].direction


@dataclass
class State:

    position: Tuple[int, int]
    arena_shape: Tuple[int, int]
    past_actions: List["Action"]
    turn: int
    points: int

    def as_tensor(self) -> np.ndarray:
        packed = list(State._pack(astuple(self)))
        return np.asarray(packed)

    @staticmethod
    def tensor_length(number_of_past_actions: int):
        return 2 + 2 + 2*number_of_past_actions + 1 + 1

    @staticmethod
    def _pack(iterable: Iterable) -> Iterable:
        for element in iterable:
            if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
                yield from State._pack(element)
            elif isinstance(element, Richtung):
                yield directions.index(element) + 1
            elif isinstance(element, FeldZustand):
                yield field_states.index(element) + 1
            elif element is None:
                yield 0
            else:
                yield element


@dataclass
class Action:

    direction: Richtung
    result: FeldZustand
