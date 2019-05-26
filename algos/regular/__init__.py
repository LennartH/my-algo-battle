import logging

from .state_based_algorithm import StateBasedAlgorithm
from .cao import Chaos, Order

logging.basicConfig(style="{", format="{levelname: >8} - {name}: {message}", level=logging.INFO)


class ChaosAndOrder(StateBasedAlgorithm):

    def __init__(self):
        super().__init__(Chaos(0.4), Order())
