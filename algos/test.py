import time

from algo_battle.domain import FeldZustand, Richtung
from algo_battle.domain.algorithmus import Algorithmus
from algo_battle.util.builtin_algorithmen import Zufall
from util import SleepWrapper


class Dot(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        return self.richtung.drehe_nach_rechts()


class Debug(Algorithmus):

    def _gib_richtung(self, letzter_zustand: FeldZustand, zug_nummer: int, aktuelle_punkte: int) -> Richtung:
        time.sleep(10)
        return self.richtung.drehe_nach_links()


# very_fast_sleep_time = 0.001
# fast_sleep_time = 0.005
# medium_sleep_time = 0.025
# slow_sleep_time = 0.1
#
#
# class DotVeryFast(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Dot, very_fast_sleep_time)
#
#
# class DotFast(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Dot, fast_sleep_time)
#
#
# class DotMedium(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Dot, medium_sleep_time)
#
#
# class DotSlow(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Dot, slow_sleep_time)
#
#
# class ZufallVeryFast(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Zufall, very_fast_sleep_time)
#
#
# class ZufallFast(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Zufall, fast_sleep_time)
#
#
# class ZufallMedium(SleepWrapper):
#
#     def __init__(self):
#         super().__init__(Zufall, medium_sleep_time)
