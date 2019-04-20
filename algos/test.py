import time

from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus
from algo_battle.util.builtin_algorithmen import Zufall
from util import SleepWrapper


class Punkt(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self.richtung.drehe_nach_rechts()


class PunktSchnell(SleepWrapper):

    def __init__(self):
        super().__init__(Punkt, 0.005)


class PunktMittel(SleepWrapper):

    def __init__(self):
        super().__init__(Punkt, 0.025)


class PunktLangsam(SleepWrapper):

    def __init__(self):
        super().__init__(Punkt, 0.1)


class ZufallMittel(SleepWrapper):

    def __init__(self):
        super().__init__(Zufall, 0.025)
