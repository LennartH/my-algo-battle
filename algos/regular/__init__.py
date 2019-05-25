import logging

from .cao import ChaosAndOrder as CaO

logging.basicConfig(style="{", format="{levelname: >8} - {name}: {message}", level=logging.INFO)


class ChaosAndOrder(CaO):

    def __init__(self):
        super().__init__(0.4)
