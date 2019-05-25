import random

from enum import Enum, auto
from typing import Dict, Any, Callable, Iterable, Optional
from algo_battle.domain import FeldZustand, Richtung
from .base import StateBasedAlgorithm, Rect


class Mode(Enum):

    Chaos = auto()
    SeekCorner = auto()
    Order = auto()
    Search = auto()


class ChaosAndOrder(StateBasedAlgorithm):

    def __init__(self, chaos_factor: float):
        super().__init__("Lennart: Chaos und Ordnung")
        self._chaos_factor = chaos_factor

        self._min_chaos_steps = 0
        self._max_chaos_steps = 0
        self._chaos_steps = 0

        self._init_seek_corner = True
        self._seek_corner_direction1 = None
        self._seek_corner_direction2 = None
        self._seek_corner_counter = 0
        self._exit_seek_corner = False

        self._init_order = True
        self._order_change_line = False
        self._order_change_direction = None
        self._order_reline_direction = None
        self._order_found_free_cell = False
        self._exit_order_for_search = False

        self._exit_search = False

    def _get_initial_state(self) -> Any:
        return Mode.Chaos

    def _create_actions_by_state(self) -> Dict[Any, Callable[[FeldZustand, int, int], Richtung]]:
        return {
            Mode.Chaos: self._cause_chaos,
            Mode.SeekCorner: self._seek_corner,
            Mode.Order: self._bring_order,
            Mode.Search: self._search_free_cell
        }

    def _create_state_transitions(self) -> Dict[Any, Iterable[Callable[[FeldZustand, int, int], Optional[Any]]]]:
        return {
            Mode.Chaos: [
                lambda _, t, __: Mode.Order if t >= (self.arena.punkte_maximum * self._chaos_factor) else None
            ],
            # Mode.Order: [
            #     lambda *_: Mode.Search if self._exit_order_for_search else None
            # ],
            # Mode.Search: [
            #     lambda *_: Mode.Order if self._exit_search else None
            # ]
        }

    def _bereite_vor(self):
        super()._bereite_vor()

        average_arena_form = (self.arena.breite + self.arena.hoehe) / 2
        self._min_chaos_steps = average_arena_form // 10
        self._max_chaos_steps = int((1 - self._chaos_factor) * average_arena_form)

    def _cause_chaos(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if self._chaos_steps <= 0 or letzter_zustand.ist_blockiert:
            self._chaos_steps = random.randint(self._min_chaos_steps, self._max_chaos_steps)
            if letzter_zustand == FeldZustand.Wand:
                return self.richtung.gegenteil
            # TODO Improve chaos direction
            return Richtung.zufall(ausser=[self.richtung, self.richtung.gegenteil])

        self._chaos_steps -= 1
        return self.richtung

    def _seek_corner(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if self._init_seek_corner:
            self._init_seek_corner = False
            directions_with_distances = sorted(((d, self.abstand(d)) for d in Richtung), key=lambda t: t[1], reverse=True)
            self._seek_corner_direction1 = directions_with_distances[0][0]
            self._seek_corner_direction2 = directions_with_distances[1][0]
            self._seek_corner_counter = 0

        richtung = self.richtung
        if letzter_zustand.ist_blockiert:
            self._exit_seek_corner = True
        else:
            richtung = self._seek_corner_direction1 if self._exit_seek_corner % 2 == 0 else self._seek_corner_direction2
            self._seek_corner_counter += 1
        return richtung

    def _bring_order(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if self._init_order:
            self._init_order = False
            distances_by_direction = {d: self.abstand(d) for d in Richtung}
            initial_order_direction: Richtung = max(distances_by_direction.items(), key=lambda t: t[1])[0]

            left_initial_direction = initial_order_direction.drehe_nach_links()
            right_initial_direction = initial_order_direction.drehe_nach_rechts()
            self._order_change_direction: Richtung = max((
                (left_initial_direction, distances_by_direction[left_initial_direction]),
                (right_initial_direction, distances_by_direction[right_initial_direction])
            ), key=lambda t: t[1])[0]
            return initial_order_direction

        richtung = self.richtung
        self._order_found_free_cell |= letzter_zustand == FeldZustand.Frei
        if letzter_zustand.ist_blockiert and not self._order_change_line:
            self._order_change_line = True
            if not self._order_found_free_cell:
                self._order_change_direction = self._order_change_direction.gegenteil
                self._exit_order_for_search = True
            richtung = self._order_change_direction
            self._order_reline_direction = self.richtung.gegenteil
        elif self._order_change_line:
            self._order_change_line = False
            self._order_found_free_cell = False
            richtung = self._order_reline_direction
        return richtung

    def _search_free_cell(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if letzter_zustand == FeldZustand.Frei or letzter_zustand.ist_blockiert:
            self._exit_search = True
            self._exit_order_for_search = False
            self._order_found_free_cell = False
            if letzter_zustand.ist_blockiert:
                self._order_change_direction = self._order_change_direction.gegenteil

        return self.richtung
