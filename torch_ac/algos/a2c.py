import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        update_terminal_loss = 0
        update_alignment_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss
            # evaluation then happens in bulk here.
            # Passed in a tensor of |inds| x [image size]
            # the value here is redundant. only dist isn't contained in sb.
            # this is only to encourage exploration
            if self.acmodel.optlib:
                if self.acmodel.recurrent:
                    dist, value, switch, prob_out, prob_in, memory = self.acmodel(sb.obs, memory * sb.mask)
                else:
                    dist, value, switch, prob_out, prob_in = self.acmodel(sb.obs)
            else:
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                else:
                    dist, value = self.acmodel(sb.obs)
                    # retrieving again just because didn't store distribution

            # loss function uses entropy
            entropy = dist.entropy().mean()

            # NOTE: every action was chosen both because of a policy (dist) and because of a probability of
            # staying/transitioning on that symbol.

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            if self.acmodel.optlib:
                alpha = 0.01

                tl_scale = sb.returnn # previously sb.advantage
                terminal_loss = -(torch.log(prob_out * switch.squeeze() + (1-prob_out) * (1-switch.squeeze())) * tl_scale).mean()

                js = 0.5 * (prob_in * torch.log(prob_in) + prob_out * torch.log(prob_out)) - 0.5*(prob_in + prob_out) * torch.log(0.5 * (prob_in + prob_out))
                return_js = sb.returnn * js 
                probability_alignment_loss = alpha * torch.mean(return_js)
                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss + terminal_loss + probability_alignment_loss

                # TODO forced term that also tunes the termination policy?

            else:
                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()

            if self.acmodel.optlib:
                update_terminal_loss += terminal_loss.item()
                update_alignment_loss += probability_alignment_loss.item()

            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        update_terminal_loss /= self.recurrence
        update_alignment_loss /= self.recurrence
        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "alignment_loss": update_alignment_loss,
            "terminal_loss": update_terminal_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
