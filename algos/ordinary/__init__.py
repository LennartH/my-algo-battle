import logging
import time

from enum import Enum, auto
from typing import Dict, Any, Callable, Iterable, Optional, Tuple
from dataclasses import dataclass, replace
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus
from .base import StateBasedAlgorithm

logging.basicConfig(style="{", format="{levelname: >8} - {name}: {message}", level=logging.INFO)


class Mode(Enum):

    SeekCenter = auto()
    Divide = auto()
    SwitchRect = auto()


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def distance(self, position: Tuple[int, int], direction: Richtung) -> int:
        if direction is Richtung.Oben:
            return position[1] - self.y
        if direction is Richtung.Rechts:
            return self.width + self.x - position[0]
        if direction is Richtung.Unten:
            return self.height + self.y - position[1]
        if direction is Richtung.Links:
            return position[0] - self.x

    def is_on_border(self, position: Tuple[int, int]) -> bool:
        return position[0] <= self.x or position[1] <= self.y or position[0] >= self.x + self.width or position[1] >= self.y + self.height


class DivideAndConquer(StateBasedAlgorithm):

    def __init__(self):
        super().__init__("Teile und Herrsche")
        self._current_rect: Rect = None
        self._divide_directions = [Richtung.Oben, Richtung.Unten, Richtung.Links, Richtung.Rechts]
        self._blockades_hit = 0
        self._centers_reached = 0

    @property
    def position(self) -> Tuple[int, int]:
        return self._x, self._y

    def _bereite_vor(self):
        super()._bereite_vor()
        self._current_rect = Rect(0, 0, *self.arena.form)

    def _get_initial_state(self) -> Any:
        return Mode.SeekCenter

    def _create_actions_by_state(self) -> Dict[Any, Callable[[FeldZustand, int, int], Richtung]]:
        return {
            Mode.SeekCenter: self._seek_center,
            Mode.Divide: self._divide_rect,
            Mode.SwitchRect: self._switch_rect
        }

    def _create_state_transitions(self) -> Dict[Any, Iterable[Callable[[FeldZustand, int, int], Optional[Any]]]]:
        return {
            Mode.SeekCenter: [
                lambda s, *_: Mode.Divide if self._center_reached() or s == FeldZustand.Belegt else None
            ],
            Mode.Divide: [
                lambda *_: Mode.SeekCenter if self._centers_reached == 1 and self._blockades_hit == 2 else None,
                lambda *_: Mode.SwitchRect if self._blockades_hit > 3 else None
            ],
            Mode.SwitchRect: [lambda *_: Mode.SeekCenter]
        }

    def _seek_center(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        # TODO Handle blockades
        max_distance = 0
        max_direction = None
        for direction in Richtung:
            direction_distance = self._current_rect.distance(self.position, direction)
            if direction_distance >= max_distance:
                max_distance = direction_distance
                max_direction = direction
        return max_direction

    def _divide_rect(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if self._hit_blockade(letzter_zustand):
            self._blockades_hit += 1

        if self._blockades_hit > 3:
            return self.richtung.gegenteil
        else:
            return self._divide_directions[self._blockades_hit]

    def _switch_rect(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        self._blockades_hit = 0
        self._centers_reached = 0

        if self._current_rect.size == self.arena.form:
            self._current_rect = Rect(0, 0, self._current_rect.width // 2, self._current_rect.height // 2)
        elif self._current_rect.x == 0 and self._current_rect.y == 0:
            self._current_rect = replace(self._current_rect, x=self._current_rect.x + self._current_rect.width)
        elif self._current_rect.x != 0 and self._current_rect.y != 0:
            self._current_rect = replace(self._current_rect, x=0)
        else:
            self._current_rect = replace(self._current_rect, y=self._current_rect.y + self._current_rect.height)

        return self.richtung

    def _hit_blockade(self, letzter_zustand: FeldZustand):
        return letzter_zustand.ist_blockiert or self._current_rect.is_on_border(self.position)

    def _center_reached(self) -> bool:
        distances = [self._current_rect.distance(self.position, d) for d in Richtung]
        center_reached = all(abs(distances[0] - d) <= 1 for d in distances[1:])
        if center_reached:
            self._centers_reached += 1
        return center_reached


class Dot(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self.richtung.drehe_nach_rechts()


class Debug(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        time.sleep(10)
        return self.richtung.drehe_nach_links()
