from typing import Dict, Optional, Iterable
from abc import ABC, abstractmethod
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import ArenaDefinition
from util import ExtendedAlgorithm


class StateBasedAlgorithm(ExtendedAlgorithm):

    def __init__(self, initial_state: "State", *other_states, name: str = None):
        super().__init__(name)
        self._state = initial_state

        states = {initial_state}
        if len(other_states) and isinstance(other_states[0], Iterable):
            states.update(other_states[0])
        else:
            states.update(other_states)
        for state in states:
            state.algorithm = self
        self._states_by_id: Dict[str, State] = {s.id: s for s in states}

    @property
    def state(self) -> "State":
        return self._state

    def get_state(self, state_id: str) -> "State":
        return self._states_by_id[state_id]

    def _bereite_vor(self):
        self._state.on_entry()

    def _get_direction(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Richtung:
        new_state_id = self.state.get_next_state(cell_state, current_turn, current_points)
        if new_state_id:
            self.state.on_exit()
            self._state = self.get_state(new_state_id)
            self.state.on_entry()
        return self.state.get_direction(cell_state, current_turn, current_points)


class State(ABC):

    def __init__(self, id: str = None, algorithm: StateBasedAlgorithm = None):
        if id is None:
            self._id = self.__class__.__name__
        else:
            self._id = id
        self._algorithm = algorithm
        self._entry_counter = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def entry_counter(self) -> int:
        return self._entry_counter

    @property
    def algorithm(self) -> StateBasedAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: StateBasedAlgorithm):
        self._algorithm = algorithm

    @property
    def arena(self) -> ArenaDefinition:
        return self.algorithm.arena

    @property
    def direction(self) -> Richtung:
        return self.algorithm.richtung

    @property
    def already_visited_counter(self) -> int:
        return self.algorithm.already_visited_counter

    def _on_entry(self):
        self.on_entry()
        self._entry_counter += 1

    def on_entry(self):
        pass

    @abstractmethod
    def get_next_state(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Optional[str]:
        pass

    @abstractmethod
    def get_direction(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Richtung:
        pass

    def on_exit(self):
        pass
