# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Generator, Tuple, Union


class ReplayBuffer:
    """
    Fixed-size circular buffer to store state and next-state experience tuples.

    Attributes:
        states (Tensor): Buffer of current states.
        next_states (Tensor): Buffer of next states.
        buffer_size (int): Maximum number of elements in the buffer.
        device (str or torch.device): Device where tensors are stored.
        step (int): Current write index.
        num_samples (int): Total number of inserted samples (up to buffer_size).
    """

    def __init__(
        self,
        obs_dim: int,
        buffer_size: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize a ReplayBuffer object on a fixed device.

        Args:
            obs_dim (int): Dimension of the observation space.
            buffer_size (int): Maximum number of transitions to store.
            device (str or torch.device): Torch device where buffers are allocated ('cpu' or 'cuda').

        Raises:
            AssertionError: If preallocated tensors are not created on the
                requested device type.
        """
        self.device = torch.device(device)
        self.buffer_size = buffer_size

        # Pre-allocate buffers on the target device
        self.states = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=self.device
        )
        self.next_states = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=self.device
        )
        assert self.states.device.type == self.device.type, (
            f"ReplayBuffer tensors are on {self.states.device}, expected {self.device}"
        )

        self.step = 0
        self.num_samples = 0

    def insert(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> None:
        """
        Add a batch of states and next_states to the buffer.

        Args:
            states (Tensor): Batch of current states (batch_size, obs_dim).
            next_states (Tensor): Batch of next states (batch_size, obs_dim).
        """
        # Move incoming data to buffer device if necessary
        states = states.to(self.device, non_blocking=True)
        next_states = next_states.to(self.device, non_blocking=True)

        batch_size = states.shape[0]
        end = self.step + batch_size

        if end <= self.buffer_size:
            self.states[self.step : end] = states
            self.next_states[self.step : end] = next_states
        else:
            # Wrap around
            first_part = self.buffer_size - self.step
            self.states[self.step :] = states[:first_part]
            self.next_states[self.step :] = next_states[:first_part]
            remainder = batch_size - first_part
            self.states[:remainder] = states[first_part:]
            self.next_states[:remainder] = next_states[first_part:]

        # Update pointers
        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + batch_size)

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        allow_replacement: bool = True,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yield `num_mini_batch` mini‑batches of (state, next_state) tuples from the buffer,
        each of length `mini_batch_size`.

                If the total number of requested samples is larger than the number of
                items currently stored (`len(self)`), the method will

        * raise an error  when `allow_replacement=False`;
        * silently sample **with replacement** when `allow_replacement=True`
          (the default).

                For oversized sampling with replacement, ``torch.randint`` is used
                directly to avoid allocating large temporary permutations.

        Args
        ----
        num_mini_batch : int
        mini_batch_size : int
        allow_replacement : bool, optional
            Whether to allow sampling with replacement when the request
            exceeds the number of stored transitions.
        """
        total = num_mini_batch * mini_batch_size

        # Sampling with replacement may contain duplicates. This preserves the
        # same statistical behavior while avoiding large temporary allocations.
        if total > self.num_samples:
            if not allow_replacement:
                raise ValueError(
                    f"Not enough samples in buffer: requested {total}, "
                    f"but have {self.num_samples}"
                )
            indices = torch.randint(
                0,
                self.num_samples,
                (total,),
                device=self.device,
            )
        else:
            # Sample WITHOUT replacement
            indices = torch.randperm(self.num_samples, device=self.device)[:total]

        # Yield the mini‑batches
        for i in range(num_mini_batch):
            batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            yield self.states[batch_idx], self.next_states[batch_idx]

    def __len__(self) -> int:
        """
        Return the number of valid samples currently stored in the buffer.
        """
        return self.num_samples
