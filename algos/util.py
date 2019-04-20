import time

from typing import Type

from algo_battle.domain import FeldZustand, Richtung, ArenaDefinition
from algo_battle.domain.algorithmus import Algorithmus


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
