import logging

import torch
from allennlp.common import Registrable

logger = logging.getLogger(__name__)


class StateStackManager(Registrable):

    """
    """

    def __init__(
        self,
    ) -> None:
        pass

        # if self.training:
        #     self._state_stack_manager.initialize_stacks(
        #         batch_size)
        # else:
        #     self._state_stack_manager.initialize_stacks(
        #         batch_size*self._beam_size)


class LSTMStateStackManager(StateStackManager):

    """
    """

    def __init__(self) -> None:
        self._hidden_state_stacks = None
        self._context_state_stacks = None

        self._count = None

    def initialize_stacks(self, num_stacks) -> None:
        """
        We dynamically llocate `` number of dynamically

        """

        self._hidden_state_stacks = [[] for _ in range(num_stacks)]
        self._context_state_stacks = [[] for _ in range(num_stacks)]

        self._count = 0

    def push_child_states(self, batch_child_states) -> None:

        for idx, child_states in enumerate(batch_child_states):

            hidden_states, context_states = child_states

            # We need to reverse in order to from right to left
            # so that we can pop the child states from left to right
            self._hidden_state_stacks[idx].extend(hidden_states.flip(0))
            self._context_state_stacks[idx].extend(context_states.flip(0))

#         logging.info(f"{self._count} - {len(self._hidden_state_stacks)}")
        self._count += 1

    def pop_current_states(self, num_stacks, state_dim=None, device=None):
        """
        """
        # Pop the lastest state
        # shape: ()
        current_hiddens = []
        current_contexts = []
        state_mask = []

        for hidden_state_stack, context_state_stack in zip(self._hidden_state_stacks[:num_stacks],
                                                           self._context_state_stacks[:num_stacks]):
            if len(hidden_state_stack) != 0:
                current_hiddens.append(hidden_state_stack.pop())
                current_contexts.append(context_state_stack.pop())
                state_mask.append(1)
            else:
                current_hiddens.append(
                    torch.zeros(state_dim, device=device))
                current_contexts.append(
                    torch.zeros(state_dim, device=device))
                state_mask.append(0)

        current_hiddens = torch.stack(current_hiddens)
        current_contexts = torch.stack(current_contexts)
        state_mask = torch.tensor(state_mask, dtype=torch.bool, device=device)

        return current_hiddens, current_contexts, state_mask


class GRUStateStackManager(StateStackManager):

    """
    """

    def __init__(self) -> None:
        self._hidden_state_stacks = None

        self._count = None

    def initialize_stacks(self, num_stacks) -> None:
        """
        We dynamically llocate `` number of dynamically

        """

        self._hidden_state_stacks = [[] for _ in range(num_stacks)]

        self._count = 0

    def push_child_states(self, batch_child_states) -> None:

        for idx, child_states in enumerate(batch_child_states):

            hidden_states = child_states

            # We need to reverse in order to from right to left
            # so that we can pop the child states from left to right
            self._hidden_state_stacks[idx].extend(hidden_states.flip(0))

        # logging.info(f"{self._count} - {len(self._hidden_state_stacks)}")
        self._count += 1

    def pop_current_states(self, num_stacks, state_dim=None, device=None):
        """
        """
        # Pop the lastest state
        # shape: ()
        current_hiddens = []
        state_mask = []

        for hidden_state_stack in self._hidden_state_stacks[:num_stacks]:
            if len(hidden_state_stack) != 0:
                current_hiddens.append(hidden_state_stack.pop())
                state_mask.append(1)
            else:
                current_hiddens.append(
                    torch.zeros(state_dim, device=device))
                state_mask.append(0)

        current_hiddens = torch.stack(current_hiddens)
        state_mask = torch.tensor(state_mask, dtype=torch.bool, device=device)

        return current_hiddens, state_mask
