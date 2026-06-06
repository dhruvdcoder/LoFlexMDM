from typing import Any, Dict, Generator, Literal, Optional, Tuple, Union
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from xlm.harness import Harness, LRSchedulerWithConfig
import hydra
import logging

logger = logging.getLogger(__name__)




class FlexMDMVariationalHarness(Harness):
    """Variational harness for learned noising FlexMDM with two separate optimizers."""

    aux_model: Optional[nn.Module]

    def instantiate_model(self):
        """Instantiate both main model and auxiliary model."""
        self.model = hydra.utils.instantiate(self.config.model)
        if self.config.get("aux_model", None) is not None:
            self.aux_model = hydra.utils.instantiate(self.config.aux_model)
        else:
            self.aux_model = None

    def instantiate_loss_function(self):
        """Instantiate loss function and wire up model references."""
        self.loss_function = hydra.utils.instantiate(self.config.loss)
        if self.loss_function.tokenizer is None:
            self.loss_function.tokenizer = self.tokenizer
        if self.loss_function.model is None:
            self.loss_function.model = self.model
        if hasattr(self.loss_function, "aux_model"):
            if self.loss_function.aux_model is None:
                self.loss_function.aux_model = self.aux_model
        # check for consistency with the predictor
        if hasattr(self, "check_loss_predictor_consistency"):
            self.check_loss_predictor_consistency()

    def top_level_named_modules(
        self,
    ) -> Generator[Tuple[str, nn.Module], None, None]:
        """Yield both model and aux_model for optimization."""
        yield "model", self.model
        if self.aux_model is not None:
            yield "aux_model", self.aux_model

    def configure_optimizers(
        self,
    ) -> Dict[
        Literal["optimizer", "lr_scheduler"],
        Union[Optimizer, LRSchedulerWithConfig],
    ]:
        """
        Configure optimizer with proper weight decay handling for both models.
        - Main model parameters (θ) with and without weight decay
        - Aux model parameters (φ) with and without weight decay
        """
        partial_optimizer = hydra.utils.instantiate(
            self.config.optimizer, _partial_=True
        )

        groups = []

        # Handle main model parameters
        if hasattr(self.model, "get_param_groups"):
            groups.extend(self.model.get_param_groups())
        else:
            main_params_with_weight_decay = list(
                p for _, p in self.model.get_named_params_for_weight_decay()
            )
            main_params_without_weight_decay = list(
                p for _, p in self.model.get_named_params_for_no_weight_decay()
            )
            logger.info(
                f"Num params with weight decay in the `model`: {len(main_params_with_weight_decay)}"
            )
            logger.info(
                f"Num params without weight decay in the `model`: {len(main_params_without_weight_decay)}"
            )
            if main_params_with_weight_decay:
                groups.append({"params": main_params_with_weight_decay})
            if main_params_without_weight_decay:
                groups.append(
                    {
                        "params": main_params_without_weight_decay,
                        "weight_decay": 0.0,
                    }
                )

        # Handle aux model parameters if present
        if self.aux_model is not None:
            if hasattr(self.aux_model, "get_param_groups"):
                groups.extend(self.aux_model.get_param_groups())
            else:
                aux_params_with_weight_decay = list(
                    p
                    for _, p in self.aux_model.get_named_params_for_weight_decay()
                )
                aux_params_without_weight_decay = list(
                    p
                    for _, p in self.aux_model.get_named_params_for_no_weight_decay()
                )
                logger.info(
                    f"Num params with weight decay in the `aux_model`: {len(aux_params_with_weight_decay)}"
                )
                logger.info(
                    f"Num params without weight decay in the `aux_model`: {len(aux_params_without_weight_decay)}"
                )
                aux_lr = None
                if self.config.get("aux_lr", None) is not None:
                    aux_lr = self.config.aux_lr
                if aux_params_with_weight_decay:
                    temp = {"params": aux_params_with_weight_decay}
                    if aux_lr is not None:
                        temp["lr"] = aux_lr
                    groups.append(temp)
                if aux_params_without_weight_decay:
                    temp = {
                        "params": aux_params_without_weight_decay,
                        "weight_decay": 0.0,
                    }
                    if aux_lr is not None:
                        temp["lr"] = aux_lr
                    groups.append(temp)

        optimizer = partial_optimizer(groups)
        lr_scheduler: LRSchedulerWithConfig = self.create_lr_scheduler(
            optimizer, **self.config.lr_scheduler
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["train", "val", "test", "predict"],
    ) -> Dict[str, Any]:
        loss_dict = super()._step(batch, batch_idx, dataloader_idx, stage)
        if stage == "train":
            kwargs = dict(
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False,
                rank_zero_only=True,
                logger=True,
                add_dataloader_idx=False,
            )
            if loss_dict.get("loss_theta", None) is not None:
                self.log(
                    "train/loss_theta",
                    loss_dict["loss_theta"].detach().mean(),
                    **kwargs,
                )
            if loss_dict.get("advantage", None) is not None:
                temp = loss_dict["advantage"].detach().abs().max()
                self.log(
                    "train/advantage",
                    temp,
                    **kwargs,
                )
            if loss_dict.get("log_p_diff", None) is not None:
                # take largest absolute value
                temp = loss_dict["log_p_diff"].detach().abs().max()
                self.log(
                    "train/log_p_diff",
                    temp,
                    **kwargs,
                )
            if loss_dict.get("reg_loss", None) is not None:
                self.log(
                    "train/reg_loss",
                    loss_dict["reg_loss"].detach().mean(),
                    **kwargs,
                )
            if loss_dict.get("unmask_loss", None) is not None:
                self.log(
                    "train/unmask_loss",
                    loss_dict["unmask_loss"].detach().mean(),
                    **kwargs,
                )
            if loss_dict.get("insertion_loss", None) is not None:
                self.log(
                    "train/insertion_loss",
                    loss_dict["insertion_loss"].detach().mean(),
                    **kwargs,
                )
        return loss_dict

    def on_validation_start(self):
        if self.config.get("aux_analysis", False):
            self.predictor.model = self.aux_model
        super().on_validation_start()

    def on_test_start(self):
        if self.config.get("aux_analysis", False):
            self.predictor.model = self.aux_model
        super().on_test_start()


class ExtractSharedModelState:
    def extract_state_dict(self, checkpoint: dict) -> dict:
        model_state_dict = {}
        for key in checkpoint.keys():
            if key.startswith("model."):
                model_state_dict[key[6:]] = checkpoint[key]
        return model_state_dict