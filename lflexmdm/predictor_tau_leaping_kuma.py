"""Simplified Tau-Leaping Predictor for LFlexMDM.

This implements the tau-leaping sampling algorithm from the paper, using the
parameterized insertion and unmasking rates:
    - R^{θ,i}_{ins,t}(x_t^i, mask) = a^{θ,i} * α_t
    - R^{θ,i}_{unmask,t}(mask, y^i) = b^{θ,i} * β_t * K^{θ,i}(y^i | x_t)

Key features:
    - Insertions happen on the LEFT of each token
    - Initial sequence is [EOS] for unconditional, [prefix] [BOS] [EOS] for conditional
    - Unmasking and insertion are independent events that can happen simultaneously
    - When multiple unmaskings compete, sample from K (the model distribution)
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Integer, Float
from torch import Tensor as TT

from xlm.datamodule import Tokenizer
from xlm.harness import Predictor, PredictorHistoryMixin
from .schedules import FlexMDMSchedule
from xlm.utils.nn import (
    sample_from_logits,
    sample_from_top_k,
    sample_from_top_p,
    select_random_indices,
)
from xlm import flags

from .types_flexmdm import (
    FlexMDMAuxModel,
    FlexMDMAuxPredictionDict,
    FlexMDMBatch,
    FlexMDMPredictionDict,
    FlexMDMModel,
)

import time


class TauLeapingPredictor(
    torch.nn.Module,
    PredictorHistoryMixin,
    Predictor[FlexMDMBatch, FlexMDMPredictionDict],
):
    """Simplified Tau-Leaping predictor based on the paper's algorithm.

    This predictor implements the tau-leaping sampling scheme where:
    1. Insertion events add new masks to the LEFT of existing tokens
    2. Unmasking events reveal the true tokens under masks
    3. Both events can occur simultaneously and independently
    """

    def __init__(
        self,
        max_steps: int,
        max_new_tokens: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[FlexMDMModel] = None,
        noise_schedule: Optional[FlexMDMSchedule] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        confidence: Optional[
            Literal[
                "rates",
                "rates_diff",
                "prob_diff",
                "entropy",
                "top_rate",
                "top_prob",
            ]
        ] = None,
        count_1: bool = False,
        return_history: bool = False,
    ):
        """Initialize the Tau-Leaping Predictor.

        Args:
            max_steps: Maximum number of prediction steps.
            max_new_tokens: Maximum number of new tokens to generate (optional).
            tokenizer: Tokenizer for encoding/decoding.
            model: The LFlexMDM model that returns (K, b, a).
            noise_schedule: Schedule containing insertion and unmasking schedules.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            confidence: Confidence-based decoding method (None for standard tau-leaping).
            count_1: Whether to require exactly 1 event for unmasking.
            return_history: Whether to track and return generation history.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        super().__init__()
        self.init_history(return_history=return_history, decode_fn=self.decode)
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.dt = (1 - 1e-5) / (max_steps + 1)
        self.confidence = confidence
        self.count_1 = count_1
        # Set up sampling function
        if top_k is None and top_p is None:
            self.sampling_function = sample_from_logits
        elif top_k is not None and top_p is None:
            self.sampling_function = partial(sample_from_top_k, top_k)
        elif top_k is None and top_p is not None:
            self.sampling_function = partial(sample_from_top_p, top_p)
        else:
            raise ValueError("Both top_k and top_p cannot be non-None")

        self.noise_schedule = noise_schedule
        self.insertion_schedule = noise_schedule.insertion_noise_schedule
        self.unmasking_schedule = noise_schedule.unmasking_noise_schedule

    def reset(self):
        """Reset predictor state (no state to reset for this simple predictor)."""
        pass

    def decode(
        self, results: Dict[str, Any]
    ) -> Tuple[List[str], Integer[TT, " batch seq_len"]]:
        """Decode the results to text."""
        x: Integer[TT, " batch seq_len"] = results["x_t"]
        out_with_spl_tokens: List[str] = self.tokenizer.batch_decode(
            x, skip_special_tokens=True
        )
        return out_with_spl_tokens, x

    def stop(self, step_results: Dict[str, Any]) -> bool:
        """Check if we should stop generation."""
        time_ended = step_results["t"].min() >= 1.0 - 1e-5
        all_filled = not (
            (step_results["x_t"] == self.tokenizer.mask_token_id).any()
        )
        return time_ended or all_filled

    def to_dict(
        self,
        batch: FlexMDMBatch,
        preds: FlexMDMPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert predictions to a list of dictionaries."""
        from itertools import cycle

        # Format history using the mixin helper
        history_data = preds.get(
            "history", [[] for _ in range(len(preds["text"]))]
        )
        formatted_history = self.format_history_for_output(
            history_data, round_precision=4
        )
        to_zip = [
            preds["text"],
            preds["ids"].tolist(),
            formatted_history,
            preds.get("time_taken", cycle([-1])),
        ]
        metric_keys = []
        for n in preds:
            if n.startswith("metric_"):
                metric_keys.append(n)
                to_zip.append(preds[n])

        preds_list: List[Tuple[str, List[int], List[List[Any]], float]] = list(
            zip(*to_zip)
        )
        dicts: List[Dict[str, Any]] = []
        for preds_ in preds_list:
            dicts.append(
                {
                    "text": preds_[0],
                    "ids": preds_[1],
                    "history": preds_[2],
                    "time_taken": preds_[3],
                    **{k: preds_[4 + i] for i, k in enumerate(metric_keys)},
                }
            )
        return dicts

    @torch._dynamo.disable()
    def predict(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMPredictionDict:
        """Run tau-leaping generation.

        Algorithm:
            1. Start at t=0 with initial sequence (e.g., [EOS])
            2. For each step:
                a. Get model predictions: K, b, a = model(x_t, t)
                b. Compute rates: unmask_rate = b * β_t * K, insert_rate = a * α_t
                c. Sample unmaskings: if Poisson(unmask_rate * dt) > 0, sample from K
                d. Sample insertions: Poisson(insert_rate * dt) masks inserted to LEFT
                e. Both events are independent and can occur simultaneously
            3. At final step, deterministically unmask remaining masks
        """
        _start_time = time.time()

        if flags.DEBUG_PRINT_PREDS:
            with open("temp_tau_leaping.txt", "a") as f:
                f.write("-" * 100 + "\n")

        # Initialize
        xt = batch["input_ids"].clone()
        fixed_gaps = batch[
            "fixed"
        ].clone()  # 1's at fixed positions, 0's elsewhere
        attention_mask = (
            (xt != self.tokenizer.pad_token_id).bool().to(xt.device)
        )

        batch_size, max_length = xt.shape
        device = xt.device
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id

        steps = self.max_steps
        t = torch.zeros(batch_size, device=device)
        dt = self.dt

        # Initialize history tracking
        history: List[List[Tuple[str, float, int]]] = self.create_history(
            batch_size
        )
        # Record initial state (step 0)
        history = self.update_history_explicit(
            history,
            self.tokenizer.batch_decode(xt, skip_special_tokens=True),
            t.tolist(),
            0,
        )

        # Precompute indices for scatter operations
        batch_idx_L = (
            torch.arange(batch_size, device=device)
            .view(batch_size, 1)
            .expand(batch_size, max_length)
        )
        pos_idx_L = (
            torch.arange(max_length, device=device)
            .view(1, max_length)
            .expand(batch_size, max_length)
        )

        for step in range(steps):
            # region: Get model predictions: K, b, a ------------------------------
            params_theta = self.model(xt, t, attention_mask)
            vocab_logits = params_theta["vocab_logits"]
            # vocab_logits: (B, L, V) - K^{θ,i}(y | x_t)
            # params_theta contains: vocab_logits, b_ins, b_unmask, and optionally a_ins, a_unmask
            # K = softmax of vocab_logits
            K = vocab_logits.softmax(dim=-1)  # (B, L, V)
            # endregion: Get model predictions: K, b, a ------------------------------
            ############################################################################
            # region: Unmasking

            # region: Last step: deterministic unmasking, no insertions
            if step == steps - 1:
                mask_pos = xt == mask_id
                new_token = vocab_logits.argmax(dim=-1)  # Most likely token
                new_xt = xt.clone()
                new_xt[mask_pos] = new_token[mask_pos]
                new_xt = torch.where(xt == pad_id, pad_id, new_xt)
                xt = new_xt
                t = t + dt

                if flags.DEBUG_PRINT_PREDS:
                    with open("temp_tau_leaping.txt", "a") as f:
                        f.write(f"Final step t: {t[0].item():.4f}\n")
                        _decoded = self.tokenizer.batch_decode(
                            xt, skip_special_tokens=False
                        )
                        for seq in _decoded:
                            f.write(f"x: {seq}\n")
                        f.write("\n")
                continue
            # endregion: Last step: deterministic unmasking

            # Unmasking rate: b^{θ,i} * β_t * K^{θ,i}(y | x_t)
            unmasking_rate_factor = self.noise_schedule.unmasking_hazard_rate(
                t.unsqueeze(-1), params_theta
            )  # (B, L)
            unmask_rate = unmasking_rate_factor[:, :, None] * K  # (B, L, V)

            # region: Tau-leaping: Sample unmasking events
            # Sample Poisson counts for each (position, token) pair
            unmask_counts = torch.poisson(unmask_rate * dt).long()  # (B, L, V)

            # Zero out counts for non-mask positions
            unmask_counts = (
                unmask_counts
                * (xt == mask_id).unsqueeze(-1).expand_as(unmask_counts).long()
            )

            # Zero out counts for mask token itself (can't unmask to mask)
            unmask_counts[..., mask_id] = 0

            # Check if any unmasking event occurred at each position

            if self.count_1:
                unmask = unmask_counts.sum(dim=-1) == 1  # (B, L)
            else:
                unmask = unmask_counts.sum(dim=-1) > 0  # (B, L)
            if self.confidence == "rates":
                unmask = select_random_indices(
                    unmask.shape,
                    unmask.sum(dim=-1),
                    select_from_mask=(xt == mask_id),
                    selection_score=unmasking_rate_factor,
                    selection_mode="greedy",
                    score_mode="logits",
                )
            elif self.confidence == "prob_diff":
                temp = unmask_rate.softmax(dim=-1)  # (B, L, V)
                top2_probs, _ = torch.topk(temp, k=2, dim=-1)  # (B, L, 2)
                confidence = (
                    top2_probs[:, :, 0] - top2_probs[:, :, 1]
                )  # (B, L)
                unmask = select_random_indices(
                    unmask.shape,
                    unmask.sum(dim=-1),
                    select_from_mask=(xt == mask_id),
                    selection_score=confidence,
                    selection_mode="greedy",
                    score_mode="logits",
                )
            elif self.confidence == "entropy":
                temp = unmask_rate.softmax(dim=-1)  # (B, L, V)
                # confidence = -entropy
                confidence = torch.sum(
                    temp * torch.log(temp + 1e-10), dim=-1
                )  # (B, L)
                unmask = select_random_indices(
                    unmask.shape,
                    unmask.sum(dim=-1),
                    select_from_mask=(xt == mask_id),
                    selection_score=confidence,
                    selection_mode="greedy",
                    score_mode="logits",
                )
            elif self.confidence == "top_rate":
                confidence = unmask_rate.max(dim=-1)[0]  # (B, L)
                unmask = select_random_indices(
                    unmask.shape,
                    unmask.sum(dim=-1),
                    select_from_mask=(xt == mask_id),
                    selection_score=confidence,
                    selection_mode="greedy",
                    score_mode="logits",
                )
            elif self.confidence == "top_prob":
                confidence = K.max(dim=-1)[0]  # (B, L)
                unmask = select_random_indices(
                    unmask.shape,
                    unmask.sum(dim=-1),
                    select_from_mask=(xt == mask_id),
                    selection_score=confidence,
                    selection_mode="greedy",
                    score_mode="logits",
                )

            # For positions with unmask events, sample from K (competing exponentials)
            new_xt = xt.clone()
            if unmask.any():
                # Sample tokens from the model distribution K for positions with events
                sampled_tokens = self.sampling_function(vocab_logits)  # (B, L)
                new_xt = torch.where(unmask, sampled_tokens, new_xt)

            # Preserve pad tokens
            new_xt = torch.where(xt == pad_id, pad_id, new_xt)
            # Preserve already unmasked tokens (absorbing state)
            already_unmasked = (xt != mask_id) & (xt != pad_id)
            new_xt = torch.where(already_unmasked, xt, new_xt)
            # endregion: Unmasking
            ############################################################################
            ############################################################################
            # region: Insertion
            # Tau-leaping: Sample insertion events
            insertion_rate = self.noise_schedule.insertion_hazard_rate(
                t.unsqueeze(-1), params_theta
            )  # (B, L)
            insert_counts = torch.poisson(insertion_rate * dt).long()  # (B, L)

            # Get current sequence lengths
            xt_len = xt.ne(pad_id).sum(dim=1)  # (B,)

            # Zero out insertions at:
            # 1. Fixed gaps (prefix positions)
            # 2. Positions beyond current sequence length
            valid_insert_pos = pos_idx_L < xt_len.view(batch_size, 1)
            insert_counts = (
                insert_counts
                * (1 - fixed_gaps).long()
                * valid_insert_pos.long()
            )

            # Check if total insertions would exceed max_length
            total_inserts = insert_counts.sum(dim=1)  # (B,)
            valid_batch = (xt_len + total_inserts) <= max_length
            insert_counts = (
                insert_counts * valid_batch.view(batch_size, 1).long()
            )

            # Recompute after validation
            total_inserts = insert_counts.sum(dim=1)

            # --- Apply insertions: shift tokens right and insert masks on LEFT ---
            # Compute cumulative insertions (prefix sum)
            insert_cumsum = insert_counts.cumsum(dim=1)  # (B, L)
            new_len = xt_len + total_inserts  # (B,)

            # Initialize new sequence with pads, then fill with masks up to new_len
            xt_new = torch.full_like(xt, pad_id)
            mask_positions = pos_idx_L < new_len.view(batch_size, 1)
            xt_new[mask_positions] = mask_id

            # Scatter original tokens to their new positions (shifted right)
            new_positions = pos_idx_L + insert_cumsum  # (B, L)
            orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
            flat_batch = batch_idx_L[orig_mask]
            flat_pos = new_positions[orig_mask]
            xt_new[flat_batch, flat_pos] = new_xt[orig_mask]
            # endregion: Insertion
            ############################################################################

            # Update state
            xt = xt_new
            attention_mask = xt != pad_id
            t = t + dt

            # Update history after each step
            history = self.update_history_explicit(
                history,
                self.tokenizer.batch_decode(xt, skip_special_tokens=True),
                t.tolist(),
                step + 1,
            )

            if flags.DEBUG_PRINT_PREDS:
                with open("temp_tau_leaping.txt", "a") as f:
                    f.write(f"Step {step}, t: {t[0].item():.4f}\n")
                    _decoded = self.tokenizer.batch_decode(
                        xt, skip_special_tokens=False
                    )
                    for _seq_idx, seq in enumerate(_decoded):
                        f.write(f"x[{_seq_idx}]: {seq}\n")
                    f.write("\n")

        # Decode final output
        out = self.tokenizer.batch_decode(xt, skip_special_tokens=True)

        _end_time = time.time()
        _time_taken = _end_time - _start_time

        self.reset()
        return {
            "text": out,
            "ids": xt,
            "loss": None,
            "time_taken": [_time_taken] * len(out),
            "history": history,
        }


class AuxPredictor(
    torch.nn.Module,
    Predictor[FlexMDMBatch, FlexMDMAuxPredictionDict],
):
    """Predictor for FlexMDM aux model. Takes predictions from the aux model for analysis only."""

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        model: Optional[FlexMDMAuxModel] = None,
        noise_schedule: Optional[FlexMDMSchedule] = None,
    ):
        """Initialize the Tau-Leaping Predictor.

        Args:
            tokenizer: Tokenizer for encoding/decoding.
            model: The FlexMDM aux model to use for predictions.
            noise_schedule: Schedule containing insertion and unmasking schedules.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required")

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.insertion_schedule = noise_schedule.insertion_noise_schedule
        self.unmasking_schedule = noise_schedule.unmasking_noise_schedule

    def reset(self):
        """Reset predictor state (no state to reset for this simple predictor)."""
        pass

    def decode(
        self, results: Dict[str, Any]
    ) -> Tuple[List[str], Integer[TT, " batch seq_len"]]:
        raise NotImplementedError(
            "decode is not implemented for FlexMDMAuxPredictor"
        )

    def stop(self, step_results: Dict[str, Any]) -> bool:
        raise NotImplementedError(
            "stop is not implemented for FlexMDMAuxPredictor"
        )

    def to_dict(
        self,
        batch: FlexMDMBatch,
        preds: FlexMDMAuxPredictionDict,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert predictions to a list of dictionaries."""
        input_ids = batch["input_ids"]
        res = []
        for i in range(input_ids.shape[0]):
            res_ = {}
            res_["tokens"] = self.tokenizer.convert_ids_to_tokens(
                input_ids[i].tolist()
            )
            for k, v in preds.items():
                if isinstance(v, torch.Tensor):
                    values = v[i].tolist()  # type: ignore
                    # if len(values) > 0 and isinstance(
                    #    values[0], float
                    # ):  # round to 4 decimal places
                    #    values = [round(v, 4) for v in values]
                    res_[k] = values
                else:
                    res_[k] = v
            res.append(res_)
        return res

    @torch._dynamo.disable()
    def predict(
        self,
        batch: FlexMDMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> FlexMDMAuxPredictionDict:
        """Run FlexMDM aux model prediction."""

        z_1 = batch["input_ids"].clone()
        batch_size = z_1.shape[0]
        t = torch.rand(batch_size, device=z_1.device)
        attention_mask = (z_1 != self.tokenizer.pad_token_id).bool()
        params_phi = self.model(z_1, t, attention_mask)  # (B, L), (B, L)
        return {
            "attention_mask": attention_mask,
            **params_phi,
        }
