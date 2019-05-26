import random

from typing import Optional
from algo_battle.domain import FeldZustand, Richtung
from .state_based_algorithm import State


class Chaos(State):

    def __init__(self, chaos_factor: float):
        super().__init__()
        self._chaos_factor = chaos_factor

        self._min_chaos_steps = 0
        self._max_chaos_steps = 0
        self._chaos_steps = 0

    def on_entry(self):
        if self.entry_counter == 0:
            average_arena_form = (self.arena.breite + self.arena.hoehe) / 2
            self._min_chaos_steps = average_arena_form // 10
            self._max_chaos_steps = int((1 - self._chaos_factor) * average_arena_form)

    def get_next_state(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Optional[str]:
        if current_turn >= self.arena.punkte_maximum * self._chaos_factor:
            return Order.__name__
        else:
            return None

    def get_direction(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Richtung:
        change_direction = self._chaos_steps <= 0 or cell_state.ist_blockiert
        if self.already_visited_counter >= self._min_chaos_steps:
            self.algorithm._already_visited_counter = 0
            change_direction = True

        if change_direction:
            self._chaos_steps = random.randint(self._min_chaos_steps, self._max_chaos_steps)
            if cell_state == FeldZustand.Wand:
                return self.direction.gegenteil
            # TODO Improve chaos direction
            return Richtung.zufall(ausser=[self.direction, self.direction.gegenteil])

        self._chaos_steps -= 1
        return self.direction


class Order(State):

    def __init__(self):
        super().__init__()
        self._change_line = False
        self._initial_direction = None
        self._change_direction = None
        self._reline_direction = None
        self._found_free_cell = False
        self._order_for_search = False

    def on_entry(self):
        if self.entry_counter == 0:
            distances_by_direction = {d: self.algorithm.abstand(d) for d in Richtung}
            self._initial_direction = max(distances_by_direction.items(), key=lambda t: t[1])[0]

            left_initial_direction = self._initial_direction.drehe_nach_links()
            right_initial_direction = self._initial_direction.drehe_nach_rechts()
            self._change_direction: Richtung = max((
                (left_initial_direction, distances_by_direction[left_initial_direction]),
                (right_initial_direction, distances_by_direction[right_initial_direction])
            ), key=lambda t: t[1])[0]
            self.algorithm._richtung = self._initial_direction

    def get_next_state(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Optional[str]:
        return None

    def get_direction(self, cell_state: FeldZustand, current_turn: int, current_points: int) -> Richtung:
        direction = self.direction
        self._found_free_cell |= cell_state == FeldZustand.Frei
        if cell_state.ist_blockiert and not self._change_line:
            self._change_line = True
            if not self._found_free_cell:
                self._change_direction = self._change_direction.gegenteil
                self._order_for_search = True
            direction = self._change_direction
            self._reline_direction = self.direction.gegenteil
        elif self._change_line:
            self._change_line = False
            self._found_free_cell = False
            direction = self._reline_direction
        return direction
