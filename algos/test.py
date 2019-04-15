import time

from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus


sleep_time = 0


class Punkt(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        if sleep_time > 0:
            time.sleep(sleep_time)
        return self.richtung.drehe_nach_rechts()