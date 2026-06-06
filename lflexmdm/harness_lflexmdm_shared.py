"""
Harness for FlexMDM with shared transformer backbone.

This harness extends FlexMDMVariationalHarness to support a shared backbone
between the main model and auxiliary model (both receive training signals).
"""

from typing import Dict, Generator, Literal, Optional, Tuple, Union

import torch.nn as nn
from torch.optim.optimizer import Optimizer
import hydra
import logging

from lflexmdm.harness_lflexmdm import FlexMDMVariationalHarness
from xlm.harness import LRSchedulerWithConfig

logger = logging.getLogger(__name__)


class FlexMDMSharedBackboneHarness(FlexMDMVariationalHarness):
    """
    Harness for FlexMDM with shared transformer backbone.

    This harness:
    1. Instantiates a shared backbone
    2. Passes the backbone to both main model and aux model
    3. Configures optimizers with proper parameter groups:
       - Backbone params (updated by both main and aux model gradients)
       - Main model head params
       - Aux model head params
    """

    backbone: nn.Module

    def instantiate_model(self):
        """
        Instantiate shared backbone, main model, and auxiliary model.

        The backbone is instantiated first, then passed to both models.
        """
        # 1. Instantiate shared backbone
        self.backbone = hydra.utils.instantiate(self.config.backbone)
        logger.info(
            f"Instantiated shared backbone with {sum(p.numel() for p in self.backbone.parameters()):,} parameters"
        )

        # 2. Instantiate main model with backbone reference
        self.model = hydra.utils.instantiate(
            self.config.model, backbone=self.backbone
        )
        logger.info(
            f"Instantiated main model with {sum(p.numel() for p in self.model.parameters()):,} total parameters"
        )

        # 3. Instantiate aux model with backbone reference
        if self.config.get("aux_model", None) is not None:
            self.aux_model = hydra.utils.instantiate(
                self.config.aux_model, backbone=self.backbone
            )
            logger.info(
                f"Instantiated aux model with {sum(p.numel() for p in self.aux_model.parameters()):,} total parameters"
            )
        else:
            self.aux_model = None

    def top_level_named_modules(
        self,
    ) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Yield backbone, model, and aux_model as top-level modules.

        This is used for checkpointing and other operations that need
        to iterate over all top-level modules.
        """
        yield "backbone", self.backbone
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
        Configure optimizer with proper parameter groups for shared backbone.

        Parameter groups:
        1. Backbone params with weight decay
        2. Backbone params without weight decay (biases, norms)
        3. Main model head params with weight decay
        4. Main model head params without weight decay
        5. Aux model head params with weight decay
        6. Aux model head params without weight decay
        """
        partial_optimizer = hydra.utils.instantiate(
            self.config.optimizer, _partial_=True
        )

        groups = []

        # 1. Backbone parameters (updated by main model gradients)
        backbone_params_with_wd = list(
            p for _, p in self.backbone.get_named_params_for_weight_decay()
        )
        backbone_params_without_wd = list(
            p for _, p in self.backbone.get_named_params_for_no_weight_decay()
        )
        logger.info(
            f"Backbone params with weight decay: {len(backbone_params_with_wd)}"
        )
        logger.info(
            f"Backbone params without weight decay: {len(backbone_params_without_wd)}"
        )
        if backbone_params_with_wd:
            groups.append({"params": backbone_params_with_wd})
        if backbone_params_without_wd:
            groups.append(
                {"params": backbone_params_without_wd, "weight_decay": 0.0}
            )

        # 2. Main model head parameters (excludes backbone via get_named_params_*)
        main_params_with_wd = list(
            p for _, p in self.model.get_named_params_for_weight_decay()
        )
        main_params_without_wd = list(
            p for _, p in self.model.get_named_params_for_no_weight_decay()
        )
        logger.info(
            f"Main model head params with weight decay: {len(main_params_with_wd)}"
        )
        logger.info(
            f"Main model head params without weight decay: {len(main_params_without_wd)}"
        )
        if main_params_with_wd:
            groups.append({"params": main_params_with_wd})
        if main_params_without_wd:
            groups.append(
                {"params": main_params_without_wd, "weight_decay": 0.0}
            )

        # 3. Aux model head parameters (excludes backbone via get_named_params_*)
        if self.aux_model is not None:
            aux_params_with_wd = list(
                p
                for _, p in self.aux_model.get_named_params_for_weight_decay()
            )
            aux_params_without_wd = list(
                p
                for _, p in self.aux_model.get_named_params_for_no_weight_decay()
            )
            logger.info(
                f"Aux model head params with weight decay: {len(aux_params_with_wd)}"
            )
            logger.info(
                f"Aux model head params without weight decay: {len(aux_params_without_wd)}"
            )
            aux_lr = None
            if self.config.get("aux_lr", None) is not None:
                aux_lr = self.config.aux_lr
            if aux_params_with_wd:
                temp = {"params": aux_params_with_wd}
                if aux_lr is not None:
                    temp["lr"] = aux_lr
                groups.append(temp)
            if aux_params_without_wd:
                temp = {
                    "params": aux_params_without_wd,
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
