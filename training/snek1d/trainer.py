import os
import torch
import random
import logging.config
import yaml
import time
import algo_battle.util
import datetime as dt
import test

from algo_battle.domain import ArenaDefinition, FeldZustand
from algo_battle.domain.wettkampf import Wettkampf
from algo_battle.domain.util import EventStatistiken
from snek.base import directions, direction_to_action
from snek.snek1d import Snek1D, Snek1DModel, Movement, Snek1DState
from training.memory import Memory
from training.optimizer import Optimizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_path = ""
    kernel_size = 10

    policy_model = Snek1DModel(in_channels=Movement.size(), kernel_size=kernel_size, out_features=len(directions)).to(device)
    if model_state_path and os.path.isfile(model_state_path):
        logger().info(f"Loading model state from {model_state_path}")
        policy_model.load_state_dict(torch.load(model_state_path))
    policy_model.eval()

    target_model = Snek1DModel(in_channels=Movement.size(), kernel_size=kernel_size, out_features=len(directions)).to(device)
    target_model.load_state_dict(policy_model.state_dict())
    target_model.eval()

    arena_definition = ArenaDefinition(100, 100)
    matches_to_keep_in_memory = 5
    matches_till_target_update = 2
    matches_till_model_save = 10

    memory = Memory(arena_definition.punkte_maximum * matches_to_keep_in_memory)
    optimizer = Optimizer(memory, policy_model, target_model, device, gamma=0.995)

    number_of_matches = 25
    optimizer_batch_size = 512
    time_between_optimizations = 0.2
    # possible_competitors = [test.Debug]
    possible_competitors = [test.DotVeryFast]
    training_stats = EventStatistiken()
    for match_number in range(1, number_of_matches + 1):
        logger().info(f"Starting match {match_number}")
        match = Wettkampf(
            arena_definition.punkte_maximum, arena_definition,
            [TrainableSnek1D(policy_model, memory), random.choice(possible_competitors)()]
        )

        match.start()
        while match.laeuft_noch:
            logger().debug(f"Turn {match.aktueller_zug}")
            if len(memory) > optimizer_batch_size:
                with match.zug_berechtigung:
                    optimizer.execute(optimizer_batch_size)
            time.sleep(time_between_optimizations)

        match.berechne_punkte_neu()
        training_stats.speicher_runde(match)
        logger().info(f"Match {match_number} finished\n{algo_battle.util.wettkampf_ergebnis(match)}\n")

        if match_number % matches_till_target_update == 0:
            logger().info("Updating target model")
            target_model.load_state_dict(policy_model.state_dict())
        if match_number % matches_till_model_save == 0:
            save_training(policy_model, training_stats)

    if number_of_matches % matches_till_model_save != 0:
        save_training(policy_model, training_stats)
    logger().info(f"Training complete\n{training_stats.zusammenfassung}\n{training_stats.daten}")


def save_training(policy_model: Snek1DModel, training_stats: EventStatistiken, directory="models"):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"snek1d_{timestamp}"
    logger().info(f"Storing training in {os.path.join(directory, file_name)} as .pth and .csv")
    torch.save(policy_model.state_dict(), os.path.join(directory, f"{file_name}.pth"))
    training_stats.daten.to_csv(os.path.join(directory, f"{file_name}.csv"))


class TrainableSnek1D(Snek1D):

    def __init__(self, model: Snek1DModel, memory: Memory):
        super().__init__(model)
        self._memory = memory

    def _update_state(self, action_result: FeldZustand, turn: int, points: int):
        state_tensor = self._state.as_tensor(self._device)
        action = torch.tensor([[direction_to_action(self.richtung)]], device=self._device)
        super()._update_state(action_result, turn, points)
        next_state_tensor = self._state.as_tensor(self._device)
        reward = torch.tensor([self._calculate_reward()], dtype=torch.float, device=self._device)
        self._memory.append(state_tensor, action, next_state_tensor, reward)

    def _calculate_reward(self) -> float:
        state: Snek1DState = self._state
        latest_result = state.past_movements[-1].result

        if latest_result == FeldZustand.Frei:
            return 1
        if latest_result == FeldZustand.Besucht:
            return -0.1

        movement = state.past_movements[-2]
        reward = -0.1 if movement is None or movement.result.ist_betretbar else -2
        index = -3
        while state.past_movements[index] is not None and state.past_movements[index].result.ist_blockiert:
            reward *= 2
            index -= 1
        return reward


def logger():
    return logging.getLogger("Training")


if __name__ == "__main__":
    with open("..\\logging_config.yml") as f:
        logging.config.dictConfig(yaml.full_load(f))
    main()
