import time

from typing import Type, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from algo_battle.domain import FeldZustand, Richtung, ArenaDefinition
from algo_battle.domain.algorithmus import Algorithmus


class ExtendedAlgorithm(Algorithmus, ABC):

    def __init__(self, name: str = None):
        super().__init__(name)
        self._already_visited_counter = 0

    @property
    def position(self) -> "Point":
        return Point(self._x, self._y)

    @property
    def already_visited_counter(self) -> int:
        return self._already_visited_counter

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        self._already_visited_counter += 1
        if letzter_zustand == FeldZustand.Frei:
            self._already_visited_counter = 0
        return self._get_direction(letzter_zustand, zug_nummer, aktuelle_punkte)

    @abstractmethod
    def _get_direction(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Richtung:
        pass


@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other):
        if isinstance(other, int):
            return Point(self.x + other, self.y + other)
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        raise NotImplemented(f"Can't add objects of type {type(other)}")

    def __iadd__(self, other):
        if not isinstance(other, (int, Point)):
            raise NotImplemented(f"Can't add objects of type {type(other)}")

        if isinstance(other, int):
            self.x += other
            self.y += other
        elif isinstance(other, Point):
            self.x += other.x
            self.y += other.y
        return self

    def __sub__(self, other):
        if isinstance(other, int):
            return Point(self.x - other, self.y - other)
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        raise NotImplemented(f"Can't subtract objects of type {type(other)}")

    def __isub__(self, other):
        if not isinstance(other, (int, Point)):
            raise NotImplemented(f"Can't subtract objects of type {type(other)}")

        if isinstance(other, int):
            self.x -= other
            self.y -= other
        elif isinstance(other, Point):
            self.x -= other.x
            self.y -= other.y
        return self

    def __abs__(self):
        return Point(abs(self.x), abs(self.y))


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    @property
    def center(self) -> Point:
        return Point((self.x + self.width) // 2, (self.y + self.height) // 2)

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def distances(self, position: Point) -> List[Tuple[int, Richtung]]:
        return [(self.distance(position, direction), direction) for direction in Richtung]

    def distance(self, position: Point, direction: Richtung) -> int:
        if direction is Richtung.Oben:
            return position.y - self.y
        if direction is Richtung.Rechts:
            return self.width + self.x - position.x
        if direction is Richtung.Unten:
            return self.height + self.y - position.y
        if direction is Richtung.Links:
            return position.x - self.x

    def is_on_border(self, position: Point) -> bool:
        return position.x <= self.x or position.y <= self.y or position.x >= self.x + self.width or position.y >= self.y + self.height

    def direction_to_center(self, position: Point) -> List[Richtung]:
        # TODO
        pass


class SleepWrapper(Algorithmus):

    def __init__(self, algo_type: Type[Algorithmus], sleep_time: float):
        self._algo = algo_type()
        super(SleepWrapper, self).__init__(name=f"{self._algo.name} ({sleep_time})")
        self._sleep_time = sleep_time

    @property
    def richtung(self) -> Richtung:
        return self._algo.richtung

    @property
    def arena(self) -> ArenaDefinition:
        return self._algo.arena

    @arena.setter
    def arena(self, arena: ArenaDefinition):
        self._algo.arena = arena

    def abstand(self, richtung: Richtung):
        return self._algo.abstand(richtung)

    def start(self, x: int, y: int):
        self._algo.start(x, y)

    def aktualisiere(self, x: int, y: int, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int):
        time.sleep(self._sleep_time)
        self._algo.aktualisiere(x, y, letzter_zustand, zug_nummer, aktuelle_punkte)

    def _bereite_vor(self):
        self._algo._bereite_vor()

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self._algo._gib_richtung(letzter_zustand, zug_nummer, aktuelle_punkte)
