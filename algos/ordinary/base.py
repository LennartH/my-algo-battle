from typing import Any, Dict, Callable, Optional, Iterable
from abc import abstractmethod
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus


class StateBasedAlgorithm(Algorithmus):

    # noinspection PyTypeChecker
    def __init__(self, name: str = None):
        super().__init__(name)
        self._state: Any = None
        self._actions_by_state: Dict[Any, Callable[[FeldZustand, int, int], Richtung]] = None
        self._state_transitions: Dict[Any, Iterable[Callable[[FeldZustand, int, int], Optional[Any]]]] = None

    def _bereite_vor(self):
        self._state = self._get_initial_state()
        self._actions_by_state = self._create_actions_by_state()
        self._state_transitions = self._create_state_transitions()

    @abstractmethod
    def _get_initial_state(self) -> Any:
        pass

    @abstractmethod
    def _create_actions_by_state(self) -> Dict[Any, Callable[[FeldZustand, int, int], Richtung]]:
        pass

    @abstractmethod
    def _create_state_transitions(self) -> Dict[Any, Iterable[Callable[[FeldZustand, int, int], Optional[Any]]]]:
        pass

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        transitions = self._state_transitions.get(self._state)
        if transitions:
            for transition in transitions:
                new_state = transition(letzter_zustand, zug_nummer, aktuelle_punkte)
                if new_state:
                    self._state = new_state

        action = self._actions_by_state.get(self._state)
        if action:
            return action(letzter_zustand, zug_nummer, aktuelle_punkte)
        else:
            return self.richtung
