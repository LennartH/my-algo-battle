import logging
import time

from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus
from .dac import DivideAndConquer as DaQ

logging.basicConfig(style="{", format="{levelname: >8} - {name}: {message}", level=logging.INFO)


class Dot(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self.richtung.drehe_nach_rechts()


class Debug(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        time.sleep(10)
        return self.richtung.drehe_nach_links()


class DivideAndConquer(DaQ):
    pass
