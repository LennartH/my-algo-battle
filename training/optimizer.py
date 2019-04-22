import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from training.memory import Memory


class Optimizer:

    def __init__(self, memory: Memory, policy_model: nn.Module, target_model: nn.Module, device, gamma: float):
        self._memory = memory
        self._policy_model = policy_model
        self._target_model = target_model
        self._device = device
        self._gamma = gamma
        self._optimizer = optim.RMSprop(policy_model.parameters())

    def execute(self, batch_size: int):
        entries = self._memory.sample(batch_size)
        state_batch, action_batch, next_state_batch, reward_batch = zip(*entries)
        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=self._device, dtype=torch.uint8)
        non_final_next_states = torch.cat(tuple(s for s in next_state_batch if s is not None))
        reward_batch = torch.cat(reward_batch)

        state_predictions = self._policy_model(state_batch)
        state_values = state_predictions.gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self._device)
        next_state_values[non_final_mask] = self._target_model(non_final_next_states).max(1)[0].detach()
        expected_values = ((next_state_values * self._gamma) + reward_batch).unsqueeze(1)

        self._optimizer.zero_grad()
        loss = F.smooth_l1_loss(state_values, expected_values)
        loss.backward()
        for param in self._policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()


def logger():
    return logging.getLogger("Optimizer")
