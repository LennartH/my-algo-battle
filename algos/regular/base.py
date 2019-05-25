from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional, Iterable, Tuple, List
from abc import abstractmethod
from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus


# TODO Own class for States
class StateBasedAlgorithm(Algorithmus):

    # noinspection PyTypeChecker
    def __init__(self, name: str = None):
        super().__init__(name)
        self._state: Any = None
        self._actions_by_state: Dict[Any, Callable[[FeldZustand, int, int], Richtung]] = None
        self._state_transitions: Dict[Any, Iterable[Callable[[FeldZustand, int, int], Optional[Any]]]] = None

    @property
    def position(self) -> "Point":
        return Point(self._x, self._y)

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
class Rect:
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
