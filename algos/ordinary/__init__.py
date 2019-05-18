import logging

from enum import Enum, auto
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus

logging.basicConfig(style="{", format="{levelname: >8} - {name}: {message}", level=logging.INFO)


_directions = list(Richtung)


class Mode(Enum):

    SeekCenter = auto()
    Divide = auto()


class DivideAndConquer(Algorithmus):

    def __init__(self):
        super().__init__("Teile und Herrsche")
        self._current_rect = None
        self._mode = Mode.SeekCenter
        self._actions_by_mode = {
            Mode.SeekCenter: self._seek_center,
            Mode.Divide: self._divide_rect
        }

    def _bereite_vor(self):
        self._current_rect = (0, 0, *self.arena.form)

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self._actions_by_mode[self._mode](letzter_zustand, zug_nummer, aktuelle_punkte)

    def _seek_center(self, *args) -> Richtung:
        # TODO Handle blockades
        distances = [self._distance_in_rect(d) for d in _directions]
        next_direction = _directions[distances.index(max(distances))]
        if self._center_reached():
            self._mode = Mode.Divide
        return next_direction

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
