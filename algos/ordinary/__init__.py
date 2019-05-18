import logging

from enum import Enum, auto
from typing import Dict, Any, Callable, Iterable, Optional

from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus
from .base import StateBasedAlgorithm

logging.basicConfig(style="{", format="{levelname: >8} - {name}: {message}", level=logging.INFO)


_directions = list(Richtung)


class Mode(Enum):

    SeekCenter = auto()
    Divide = auto()


class DivideAndConquer(StateBasedAlgorithm):

    def __init__(self):
        super().__init__("Teile und Herrsche")
        self._current_rect = None

    def _bereite_vor(self):
        super()._bereite_vor()
        self._current_rect = (0, 0, *self.arena.form)

    def _get_initial_state(self) -> Any:
        return Mode.SeekCenter

    def _create_actions_by_state(self) -> Dict[Any, Callable[[FeldZustand, int, int], Richtung]]:
        return {
            Mode.SeekCenter: self._seek_center,
            Mode.Divide: self._divide_rect
        }

    def _create_state_transitions(self) -> Dict[Any, Iterable[Callable[[FeldZustand, int, int], Optional[Any]]]]:
        return {
            Mode.SeekCenter: [
                lambda s, t, p: Mode.Divide if self._center_reached() else None
            ]
        }

    def _seek_center(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        # TODO Handle blockades
        distances = [self._distance_in_rect(d) for d in _directions]
        return _directions[distances.index(max(distances))]

    def _divide_rect(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if letzter_zustand.ist_blockiert:
            return self.richtung.gegenteil
        return None

    def _center_reached(self) -> bool:
        distances = [self._distance_in_rect(d) for d in _directions]
        reference = distances[0]
        for distance in distances[1:]:
            if abs(reference - distance) > 1:
                return False
        return True

    def _distance_in_rect(self, direction: Richtung) -> int:
        if direction is Richtung.Oben:
            return self._y - self._current_rect[1]
        if direction is Richtung.Rechts:
            return self._current_rect[2] - self._x
        if direction is Richtung.Unten:
            return self._current_rect[3] - self._y
        if direction is Richtung.Links:
            return self._x - self._current_rect[0]


class Dot(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self.richtung.drehe_nach_rechts()
