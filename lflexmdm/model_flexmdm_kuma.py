# v2
import copy
import logging
import math
from typing import List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from torch.nn.attention import SDPBackend
from xlm.modules.position import RotaryEmbedding

logger = logging.getLogger(__name__)

# DTYPE = get_autocast_dtype()


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNormAndScale(nn.Module):
    """Performs normalization and just scaling (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: the dimension of the input.
        """
        super().__init__()
        self.norm = nn.Parameter(
            torch.ones([dim])
        )  # name is norm so that weight decay doesn't apply
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).

        Returns:
            the normalized and scaled output tensor of shape (bsz, seq_len, dim).
        """
        with torch.autocast(device_type="cuda", enabled=False):
            x = F.layer_norm(x, [self.dim], eps=self.eps)
        return x * self.norm[None, None, :]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
    ):
        """
        Args:
            hidden_size: The size of the hidden layer and the output of MLP.
            frequency_embedding_size: The size of the frequency embedding layer.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        )  # shape (frequency_embedding_size // 2,)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds scalar timesteps into vector representations.

        Args:
            t: A 1-D Tensor of bsz indices, one per batch element. These may be fractional.

        Returns:
            An (bsz, hidden_size) Tensor of positional embeddings.
        """
        args = (
            t[:, None].to(dtype=self.freqs.dtype) * self.freqs[None]
        )  # shape (bsz, dim // 2)
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)], dim=-1
        )  # shape (bsz, frequency_embedding_size)
        t_rep = self.mlp(embedding)  # shape (bsz, hidden_size)
        return t_rep


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (int): The number of classes.
        cond_size (int): The size of the conditioning input.
        label_dropout (Optional[float]): The dropout rate for class labels during training.

    Attributes:
        embedding_table (nn.Embedding): The embedding table for class labels.
        num_classes (int): The number of classes.

    """

    def __init__(
        self,
        num_classes: int,
        cond_size: int,
        label_dropout: Optional[float] = None,
    ):
        super().__init__()
        # have a special embedding at the end to represent absence of a label,
        # which will be used when a training label is dropped out
        assert label_dropout is None or 0 <= label_dropout < 1
        n = num_classes + 1 if label_dropout is not None else num_classes
        self.embedding = nn.Embedding(n, cond_size)
        self.num_classes = num_classes
        self.label_dropout = label_dropout
        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Drop out class labels during training.

        Args:
            labels (torch.Tensor): The input tensor of class labels of shape (bsz,).

        Returns:
            torch.Tensor: The modified class labels with some labels dropped by setting to the missing (last label).
        """
        if self.label_dropout is not None and self.training:
            mask = torch.rand_like(labels.float()) < self.label_dropout
            # set the dropped labels to the last class that represents absence of a label
            labels = torch.where(mask, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LabelEmbedder module.

        Args:
            labels (torch.Tensor): The input tensor of class labels of shape (bsz,).

        Returns:
            torch.Tensor: The embedded vector representations of the class labels.

        """
        labels = self.drop_labels(labels)
        embeddings = self.embedding(labels)
        return embeddings


class AdaLNModulations(nn.Module):
    """
    Produces the modulation parameters for AdaLN.
    """

    def __init__(
        self, cond_dim: int, dim: int, num_modulation_parameters: int = 6
    ):
        """
        Initializes the AdaLNModulations module.

        Args:
            cond_dim (int): The dimension of the conditioning input.
            dim (int): The hidden size.
        """
        super().__init__()
        self.num_modulation_parameters = num_modulation_parameters
        self.modulation = nn.Linear(
            cond_dim, num_modulation_parameters * dim, bias=True
        )
        self.modulation.weight.data.zero_()
        self.modulation.bias.data.zero_()

    def forward(self, c: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the AdaLNModulations module.

        Args:
            c (torch.Tensor): The conditioning input tensor.

        Returns:
            Tuple[torch.Tensor]: The modulation parameters for AdaLN.
                Each tensor has shape (bsz, 1, dim). When num_modulation_paramters=6, these tensors stand for
                the shift and scale parameters for the MHA and MLP layers, and the gating parameters:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        """
        # Apply the linear layer to get output of shape (bsz, 6 * dim).
        # Then add one dimension to the output to get shape (bsz, 1, 6 * dim).
        # Finally, chunk the output into 6 tensors of shape (bsz, 1, dim).
        return self.modulation(c)[:, None].chunk(
            self.num_modulation_parameters, dim=2
        )

    # add jit.script to make it faster ?
    @staticmethod
    def ada_ln_modulate(
        x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies adaLN modulation to the input tensor.

        Args:
            x: The input tensor of shape (bsz, seq_len, dim).
            shift: The shift parameter tensor of shape (bsz, 1, dim).
            scale: The scale parameter tensor of shape (bsz, 1, dim).

        Returns:
            The modulated output tensor of shape (bsz, seq_len, dim).
        """
        return x * (1.0 + scale) + shift


def add_bias_apply_dropout_scale(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    Adds bias, applies dropout, scales, and adds residual.

    TODO: Consider creating fused implementation using jit and two wrappers
    Args:
        x: The input tensor of shape (bsz, seq_len, dim).
        bias: The bias tensor of shape (bsz, 1, dim).
        dropout: The dropout rate.
        scale: The scale tensor of shape (bsz, 1, dim).
        residual: The residual tensor of shape (bsz, seq_len, dim).

    Returns:
        The output tensor of shape (bsz, seq_len, dim).
    """
    x = x + bias if bias is not None else x
    x = F.dropout(x, p=dropout, training=training) if dropout > 0.0 else x
    x = x * scale if scale is not None else x
    x = x + residual if residual is not None else x
    return x


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTLayer(nn.Module):
    """One layer of DDiT.

    It consists of a multi-head self-attention layer followed by a feedforward layer with adaLN and gating in between.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        d_cond: Optional[int] = None,
        force_flash_attn: bool = False,
    ):
        """
        Initialize the DDiTBlock.

        Args:
            d_model: the dimension of the input.
            nhead: the number of attention heads.
            d_cond: the dimension of the conditioning input.
            mlp_ratio: the ratio of the hidden size of the MLP/feedforward layer to the input size.
            dropout: the dropout rate.
        """
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        d_cond = d_cond or d_model // 2
        self.n_heads = nhead
        self.dim = d_model
        self.norm1 = LayerNormAndScale(d_model, eps=layer_norm_eps)
        self.dropout = dropout
        self.head_dim = d_model // nhead

        # self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim)
        self.rotary_emb = None

        # Single QKV projection
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = LayerNormAndScale(d_model, eps=layer_norm_eps)
        # TODO: consider using FusedMLP from flash_attn here
        if activation == "gelu":
            act = nn.GELU(approximate="tanh")
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} not supported")
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=True),
            act,
            nn.Linear(dim_feedforward, d_model, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.ada_ln_modulations = AdaLNModulations(d_cond, d_model)
        if force_flash_attn:
            self.attn_backend = [SDPBackend.FLASH_ATTENTION]
        else:
            # let torch choose
            self.attn_backend = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def set_rotary_emb(self, rotary_emb: Optional[RotaryEmbedding] = None):
        if rotary_emb is None:
            logger.info(
                "RotaryEmbedding not provided. Using default with size=head_dim"
            )
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        else:
            self.rotary_emb = rotary_emb
        self.rotary_emb_dim = rotary_emb.dim
        if self.rotary_emb_dim > self.head_dim:
            raise ValueError(
                "RotaryEmbedding dimension is greater than the head dimension."
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
            attention_mask: the attention mask of shape (bsz, seq_len), which is True for non-padding tokens.
        """
        if self.rotary_emb is None:
            raise ValueError(
                "RotaryEmbedding is not set. Call set_rotary_emb() to set it."
            )

        # modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_ln_modulations(c)
        )  # shapes: (bsz, 1, dim)

        # Apply adaLN before the attention
        x = AdaLNModulations.ada_ln_modulate(
            self.norm1(x), shift_msa, scale_msa
        )

        # Generate rotary position embeddings
        seq_len = x.shape[1]
        # rotary_pos_emb = self.rotary_emb(seq_len, x.device)

        # Project to q, k, v
        qkv = self.attn_qkv(x)  # shape (bsz, seq_len, 3 * dim)
        q, k, v = qkv.chunk(3, dim=-1)  # shape (bsz, seq_len, dim)

        # Reshape to (batch_size, n_heads, seq_len, head_dim)
        q = q.view(
            q.shape[0], q.shape[1], self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # shape (bsz, n_heads, seq_len, head_dim)
        k = k.view(
            k.shape[0], k.shape[1], self.n_heads, self.head_dim
        ).transpose(
            1, 2
        )  # shape (bsz, n_heads, seq_len, head_dim)
        v = v.view(
            v.shape[0], v.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary embeddings to q and k
        q_rotary = self.apply_rotary_pos_emb(
            q, positions
        )  # shape (bsz, n_heads, seq_len, head_dim)
        k_rotary = self.apply_rotary_pos_emb(
            k, positions
        )  # shape (bsz, n_heads, seq_len, head_dim)

        # Perform scaled dot-product attention
        # Make the attention mask broadcastable to (bsz, query_seq_len(1), key_seq_len(seq_len))
        # Note we want to broadcast (copy) along the query_seq_len dimension
        attn_mask = (
            attention_mask.unsqueeze(-2).unsqueeze(-2)
            if attention_mask is not None
            else None
        )  # shape (bsz, 1, 1, seq_len)

        # UPGRADE: The following context manager is not compile friendly
        # upto torch 2.5.1. It can be compiled with torch 2.6.0, but
        # due to the new default of `torch.load(weights_only=True)`,
        # torch 2.6.0 will not work with lightning 2.3, 2.4 or 2.5.
        # So untill lightning supports torch 2.6, we cannot use this context manager.
        with torch.nn.attention.sdpa_kernel(self.attn_backend):
            attn_output = F.scaled_dot_product_attention(
                q_rotary,
                k_rotary,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )  # shape (bsz, n_heads, seq_len, head_dim)

        # attn_output = F.scaled_dot_product_attention(
        #    q_rotary,
        #    k_rotary,
        #    v,
        #    attn_mask=attn_mask,
        #    dropout_p=self.dropout if self.training else 0.0,
        # )  # shape (bsz, n_heads, seq_len, head_dim)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], seq_len, self.dim)
        )  # shape (bsz, seq_len, dim)
        x_attn = self.o_proj(attn_output)  # shape (bsz, seq_len, dim)

        # Apply gating and residual connection
        x = add_bias_apply_dropout_scale(
            x_attn,
            bias=None,
            dropout=self.dropout,
            scale=gate_msa,
            residual=x,
            training=self.training,
        )

        # AdaLN -> MLP -> dropout -> scale -> residual
        x = add_bias_apply_dropout_scale(
            self.mlp(
                AdaLNModulations.ada_ln_modulate(
                    self.norm2(x), shift_mlp, scale_mlp
                )
            ),
            bias=None,
            dropout=self.dropout,
            scale=gate_mlp,
            residual=x,
            training=self.training,
        )

        return x

    def apply_rotary_pos_emb(
        self, x, positions: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: the input tensor of shape (batch_size, seq_len, num_heads, dim).

        Returns:
            The tensor with rotary position embeddings applied to the first dim/2 of the last dimension.
        """
        x_rope = x[
            ..., : self.rotary_emb_dim
        ]  # shape (bsz, seq_len, n_heads, dim/2)
        x_pass = x[..., self.rotary_emb_dim :]
        x_rotated = self.rotary_emb(x_rope, positions)  # type: ignore
        return torch.cat([x_rotated, x_pass], dim=-1)


class DDiTLayerList(nn.ModuleList):
    """A module list of DDiT blocks that share the rotary cache for the rotary embeddings."""

    def __init__(self, blocks: List[DDiTLayer], rotary_emb: RotaryEmbedding):

        for block in blocks:
            block.set_rotary_emb(rotary_emb)
        super().__init__(blocks)

    @classmethod
    def from_layer(
        cls, layer: DDiTLayer, num_layers: int, rotary_emb: RotaryEmbedding
    ):
        return cls(
            [copy.deepcopy(layer) for _ in range(num_layers)], rotary_emb
        )


class DDitFinalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dims: int,
        d_cond: int,
        layer_norm_eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.norm_final = LayerNormAndScale(d_model, eps=layer_norm_eps)
        self.linear = nn.Linear(d_model, out_dims, bias=use_bias)
        with torch.no_grad():
            self.linear.weight.zero_()  # zero init for absorbing diffusion
            # IMPORTANT: zero initialization will not work at all for var len
            # self.linear.weight.data.fill_(1.0)
        self.adaLN_modulation = AdaLNModulations(
            d_cond, d_model, num_modulation_parameters=2
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
        """
        shift, scale = self.adaLN_modulation(c)
        # region: DEBUG_SPARSE (remove normalization)
        x = self.adaLN_modulation.ada_ln_modulate(
            self.norm_final(x), shift, scale
        )
        # endregion DEBUG_SPARSE
        x = self.linear(x)
        return x


class DDitFinalLayerWithoutNormalization(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dims: int,
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dims, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()  # zero init for absorbing diffusion

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
        """
        x = self.linear(x)
        return x


class DDitFinalLayerForClassification(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dims: int,
        d_cond: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.norm_final = LayerNormAndScale(d_model, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=True),
            nn.Tanh(),
            nn.Linear(2 * d_model, d_model, bias=True),
        )
        self.linear = nn.Linear(d_model, out_dims, bias=use_bias)

        self.adaLN_modulation = AdaLNModulations(
            d_cond, d_model, num_modulation_parameters=3
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: the input tensor of shape (bsz, seq_len, dim).
            c: the conditioning input of shape (bsz, cond_dim).
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)
        # AdaLN -> MLP -> dropout -> scale -> residual
        x = add_bias_apply_dropout_scale(
            self.mlp(
                AdaLNModulations.ada_ln_modulate(
                    self.norm_final(x), shift_mlp, scale_mlp
                )
            ),
            bias=None,
            dropout=self.dropout,
            scale=gate_mlp,
            residual=x,
            training=self.training,
        )

        x = self.linear(x)
        return x


# length scalar head
class ScalarLengthHead(nn.Module):
    def __init__(
        self, d_model: int, normalized_len: int, cond_dim: int | None = None
    ):
        super().__init__()
        self.has_cond = cond_dim is not None
        if self.has_cond:
            self.adaLN_modulation = AdaLNModulations(
                cond_dim, d_model, num_modulation_parameters=2
            )

        self.norm = LayerNormAndScale(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()
        self.normalized_len = normalized_len

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        x_fp32 = x.float()
        c_fp32 = c.float() if (self.has_cond and c is not None) else None
        if self.has_cond and c_fp32 is not None:
            # shift, scale = self.adaLN(c_fp32)[:, None].chunk(2, dim=2)
            shift, scale = self.adaLN_modulation(c_fp32)
            x_fp32 = AdaLNModulations.ada_ln_modulate(
                self.norm(x_fp32), shift, scale
            )
        else:
            x_fp32 = self.norm(x_fp32)
        s = self.proj2(self.act(self.proj1(x_fp32)))
        out = self.softplus(s).squeeze(-1) * self.normalized_len
        return out.to(x.dtype)


# Scalar rate head for learnable noise rates (b^θ, a^θ, b^φ, a^φ)
class ScalarRateHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        cond_dim: int | None = None,
        rate_head_type: Literal["scalar", "discretized"] = "scalar",
        num_rate_bins: int = 100,
        min_rate: float = 0.1,
        max_rate: float = 10.0,
        use_mlp: bool = True,
        scalar_fn: Literal["softplus", "exp", "sigmoid"] = "softplus",
    ):
        super().__init__()
        self.has_cond = cond_dim is not None
        self.rate_head_type = rate_head_type
        self.use_mlp = use_mlp
        self.min_rate = min_rate
        self.max_rate = max_rate

        if self.has_cond:
            self.adaLN_modulation = AdaLNModulations(
                cond_dim, d_model, num_modulation_parameters=2
            )

        self.norm = LayerNormAndScale(d_model)
        self.scalar_fn = scalar_fn
        if use_mlp:
            # Two-layer MLP
            self.proj1 = nn.Linear(d_model, d_model)
            self.act = nn.GELU()
            hidden_dim = d_model
        else:
            # Single linear layer
            self.proj1 = None
            self.act = None
            hidden_dim = d_model

        if rate_head_type == "scalar":
            self.proj2 = nn.Linear(hidden_dim, 1)
            if scalar_fn == "softplus":
                with torch.no_grad():
                    self.proj2.bias.fill_(-4.0)  # so initial softplus
                self.fn = nn.Softplus()
            elif scalar_fn == "exp":
                self.fn = torch.exp
            elif scalar_fn == "sigmoid":
                self.fn = torch.sigmoid
            else:
                raise ValueError(f"Unknown scalar_fn: {scalar_fn}")
        elif rate_head_type == "discretized":
            self.proj2 = nn.Linear(hidden_dim, num_rate_bins)
            self.num_rate_bins = num_rate_bins
            # Create bin centers
            self.register_buffer(
                "bin_centers",
                torch.linspace(min_rate, max_rate, num_rate_bins),
            )
        else:
            raise ValueError(f"Unknown rate_head_type: {rate_head_type}")

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        x_fp32 = x.float()
        c_fp32 = c.float() if (self.has_cond and c is not None) else None
        if self.has_cond and c_fp32 is not None:
            shift, scale = self.adaLN_modulation(c_fp32)
            x_fp32 = AdaLNModulations.ada_ln_modulate(
                self.norm(x_fp32), shift, scale
            )
        else:
            x_fp32 = self.norm(x_fp32)

        if self.use_mlp:
            x_fp32 = self.act(self.proj1(x_fp32))

        if self.rate_head_type == "scalar":
            s = self.proj2(x_fp32)
            # Apply softplus and scale to [min_rate, max_rate] range
            if self.scalar_fn in ["softplus", "sigmoid"]:
                out = (
                    self.fn(s).squeeze(-1) * (self.max_rate - self.min_rate)
                    + self.min_rate
                )
            else:
                out = self.fn(s).squeeze(-1) + self.min_rate
        elif self.rate_head_type == "discretized":
            # Classifier over bins
            logits = self.proj2(x_fp32)  # (B, L, num_bins)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Expected value
            out = (probs * self.bin_centers).sum(dim=-1)  # (B, L)

        return out.to(x.dtype)


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        cond_dim: int | None = None,
        use_mlp: bool = True,
    ):
        super().__init__()
        self.has_cond = cond_dim is not None
        self.use_mlp = use_mlp

        if self.has_cond:
            self.adaLN_modulation = AdaLNModulations(
                cond_dim, d_model, num_modulation_parameters=2
            )

        self.norm = LayerNormAndScale(d_model)

        if use_mlp:
            # Two-layer MLP
            self.proj1 = nn.Linear(d_model, d_model)
            self.act = nn.GELU()
            hidden_dim = d_model
        else:
            # Single linear layer
            self.proj1 = None
            self.act = None
            hidden_dim = d_model

        self.proj2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        x_fp32 = x.float()
        c_fp32 = c.float() if (self.has_cond and c is not None) else None
        if self.has_cond and c_fp32 is not None:
            shift, scale = self.adaLN_modulation(c_fp32)
            x_fp32 = AdaLNModulations.ada_ln_modulate(
                self.norm(x_fp32), shift, scale
            )
        else:
            x_fp32 = self.norm(x_fp32)

        if self.use_mlp:
            x_fp32 = self.act(self.proj1(x_fp32))

        out = self.proj2(x_fp32)

        return out.to(x.dtype)  # (B, L, 1)


class Model(torch.nn.Module):

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        t: Integer[TT, " *batch"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        raise NotImplementedError

    def get_named_params_for_weight_decay(self):
        # all parameters except biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


class FlexMDMModel(Model):
    """DDiT based transformer that represents time/noise using AdaLN and uses rotary positional embeddings."""

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        d_cond: Optional[int] = None,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        rate_head_type: Literal["scalar", "discretized"] = "scalar",
        num_rate_bins: int = 1000,
        min_rate: float = 0.01,
        max_rate: float = 100.0,
        rate_head_use_mlp: bool = True,
        inner_autocast: bool = True,
        compile: bool = False,
        scalar_fn: Literal["softplus", "exp", "sigmoid"] = "softplus",
        b_um: Optional[float] = None,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.b_um = b_um
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        # TODO (neural net): Init embedding with appropriate distribution
        self.d_cond = d_cond or d_model // 2
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.sigma_map = TimestepEmbedder(self.d_cond, 256)
        encoder_layer = DDiTLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            self.d_cond,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = DDiTLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = DDitFinalLayer(
            d_model, num_embeddings, self.d_cond, layer_norm_eps
        )
        self.num_embeddings = num_embeddings

        # Rate heads for learnable noise
        self.b_ins_theta_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )
        self.b_unmask_theta_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )

        self.inner_autocast = inner_autocast
        if compile:
            for block in self.encoder:
                block.compile()
            self.output_layer.compile()
            self.b_ins_theta_head.compile()
            self.b_unmask_theta_head.compile()

    def forward(
        self,
        x: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Bool[TT, "batch seq_len"] = None,
    ) -> Tuple[
        Float[TT, "batch seq_len vocab_size"],
        Float[TT, "batch seq_len"],
        Float[TT, "batch seq_len"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The time of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
        Returns:
            vocab_logits: shape (batch, seq_len, vocab_size)
            b_theta: unmasking bias rates shape (batch, seq_len)
            a_theta: insertion rates shape (batch, seq_len)
        """
        B, L = x.shape

        attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x)  # (B, L, D)
        c = F.silu(self.sigma_map(t))
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        if self.inner_autocast:
            with torch.autocast(
                device_type="cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                for block in self.encoder:
                    x = block(x, c, attention_mask, positions)
        else:
            for block in self.encoder:
                x = block(x, c, attention_mask, positions)

        vocab_logits = self.output_layer(x, c)  # (B, L, V)
        b_ins = self.b_ins_theta_head(x, c)  # (B, L)
        b_unmask = self.b_unmask_theta_head(x, c)  # (B, L)
        
        if self.b_um is not None:
            b_unmask = torch.full_like(b_unmask, self.b_um)
        
        return {
            "vocab_logits": vocab_logits,
            "a_ins": None,
            "b_ins": b_ins,
            "a_unmask": None,
            "b_unmask": b_unmask,
        }


class FlexMDMAuxModel(Model):
    """Auxiliary model for learnable noise - predicts rates from clean sequences.

    Similar architecture to FlexMDMModel but only outputs rate heads b^φ and a^φ.
    """

    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        d_cond: Optional[int] = None,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        rate_head_type: Literal["scalar", "discretized"] = "scalar",
        num_rate_bins: int = 1000,
        min_rate: float = 0.01,
        max_rate: float = 100.0,
        rate_head_use_mlp: bool = True,
        inner_autocast: bool = True,
        compile: bool = False,
        scalar_fn: Literal["softplus", "exp"] = "softplus",
        b_um: Optional[float] = None,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.b_um = b_um
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.d_cond = d_cond or d_model // 2
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.sigma_map = TimestepEmbedder(self.d_cond, 256)
        encoder_layer = DDiTLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            self.d_cond,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = DDiTLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )

        # Only rate heads for auxiliary model
        self.lambda_ins_phi_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )
        self.lambda_unmask_phi_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )

        self.inner_autocast = inner_autocast
        self.compile = compile
        if self.compile:
            for block in self.encoder:
                block.compile()
            self.lambda_ins_phi_head.compile()
            self.lambda_unmask_phi_head.compile()

    def forward(
        self,
        z_1: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Bool[TT, "batch seq_len"] = None,
    ) -> Tuple[Float[TT, "batch seq_len"], Float[TT, "batch seq_len"]]:
        """
        Args:
            z_1: The clean input sequences of shape (batch, seq_len)
            t: The time of shape (batch,)
            attention_mask: The attention mask of shape (batch, seq_len), which is True for non-padding tokens.
        Returns:
            b_phi: unmasking bias rates shape (batch, seq_len)
            a_phi: insertion rates shape (batch, seq_len)
        """
        attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(z_1)  # (B, L, D)
        c = F.silu(self.sigma_map(t))
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

        if self.inner_autocast:
            with torch.autocast(
                device_type="cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                for block in self.encoder:
                    x = block(x, c, attention_mask, positions)
        else:
            for block in self.encoder:
                x = block(x, c, attention_mask, positions)

        b_ins = self.lambda_ins_phi_head(x, c)  # (B, L)
        b_unmask = self.lambda_unmask_phi_head(x, c)  # (B, L)

        if self.b_um is not None:
            b_unmask = torch.full_like(b_unmask, self.b_um)

        return {
            "b_ins": b_ins,
            "b_unmask": b_unmask,
            "a_ins": None,
            "a_unmask": None,
        }


#################################################################################
#                    Shared Backbone Architecture                               #
#################################################################################


class SharedTransformerBackbone(nn.Module):
    """
    Shared transformer backbone containing embed_tokens, sigma_map, and encoder.

    This backbone is shared between the main model (unfrozen) and the auxiliary
    model (frozen + LoRA). It exposes the encoder layers for layer-by-layer
    iteration with LoRA.
    """

    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        d_cond: Optional[int] = None,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_cond = d_cond or d_model // 2
        self.dim_feedforward = dim_feedforward or 4 * d_model

        # Embedding layer
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )

        # Timestep embedder
        self.sigma_map = TimestepEmbedder(self.d_cond, 256)

        # Encoder layers
        encoder_layer = DDiTLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            self.d_cond,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = DDiTLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )

    def embed_and_condition(
        self,
        x: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Bool[TT, "batch seq_len"],
    ) -> Tuple[
        Float[TT, "batch seq_len d_model"],
        Float[TT, "batch d_cond"],
        Integer[TT, "batch seq_len"],
    ]:
        """
        Embed tokens and compute conditioning.

        Returns:
            hidden_states: Embedded tokens (B, L, D)
            conditioning: Timestep conditioning (B, d_cond)
            positions: Position indices for rotary embeddings (B, L)
        """
        hidden_states = self.embed_tokens(x)  # (B, L, D)
        conditioning = F.silu(self.sigma_map(t))  # (B, d_cond)
        positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
        return hidden_states, conditioning, positions

    def forward(
        self,
        x: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Bool[TT, "batch seq_len"],
        inner_autocast: bool = True,
    ) -> Tuple[Float[TT, "batch seq_len d_model"], Float[TT, "batch d_cond"]]:
        """
        Full forward pass through the backbone.

        Returns:
            hidden_states: Final encoder output (B, L, D)
            conditioning: Timestep conditioning (B, d_cond)
        """
        attention_mask = attention_mask.to(torch.bool)
        hidden_states, conditioning, positions = self.embed_and_condition(
            x, t, attention_mask
        )

        if inner_autocast:
            with torch.autocast(
                device_type="cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                for block in self.encoder:
                    hidden_states = block(
                        hidden_states, conditioning, attention_mask, positions
                    )
        else:
            for block in self.encoder:
                hidden_states = block(
                    hidden_states, conditioning, attention_mask, positions
                )

        return hidden_states, conditioning

    def get_named_params_for_weight_decay(self):
        """Get parameters that should have weight decay applied."""
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        """Get parameters that should NOT have weight decay (biases, norms)."""
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


class FlexMDMModelShared(Model):
    """
    Main FlexMDM model that uses a shared transformer backbone.

    The backbone is passed in at initialization and used with gradients enabled
    (unfrozen) during forward pass.
    """

    def __init__(
        self,
        backbone: SharedTransformerBackbone,
        num_embeddings: int,
        d_model: int,
        d_cond: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
        rate_head_type: Literal["scalar", "discretized"] = "scalar",
        num_rate_bins: int = 1000,
        min_rate: float = 0.01,
        max_rate: float = 100.0,
        rate_head_use_mlp: bool = True,
        scalar_fn: Literal["softplus", "exp", "sigmoid"] = "softplus",
        b_um: Optional[float] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_cond = d_cond or d_model // 2
        self.num_embeddings = num_embeddings
        self.b_um = b_um

        # Output layer for vocab logits
        self.output_layer = DDitFinalLayer(
            d_model, num_embeddings, self.d_cond, layer_norm_eps
        )
        self.scalar_fn = scalar_fn

        # Rate heads for learnable noise
        self.b_ins_theta_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )
        self.b_unmask_theta_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )

    def forward(
        self,
        x: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Bool[TT, "batch seq_len"] = None,
    ) -> dict:
        """
        Forward pass using the shared backbone (with gradients).

        Returns:
            Dictionary with vocab_logits and rate parameters.
        """
        attention_mask = attention_mask.to(torch.bool)

        # Forward through shared backbone (gradients flow)
        hidden_states, conditioning = self.backbone(x, t, attention_mask)

        # Apply heads
        vocab_logits = self.output_layer(hidden_states, conditioning)
        b_ins = self.b_ins_theta_head(hidden_states, conditioning)
        b_unmask = self.b_unmask_theta_head(hidden_states, conditioning)

        if self.b_um is not None:
            b_unmask = torch.full_like(b_unmask, self.b_um)

        return {
            "vocab_logits": vocab_logits,
            "a_ins": None,
            "b_ins": b_ins,
            "a_unmask": None,
            "b_unmask": b_unmask,
        }

    def get_named_params_for_weight_decay(self):
        """Get head parameters that should have weight decay (excludes backbone)."""
        for name, param in self.named_parameters():
            # Skip backbone parameters - they're handled separately
            if name.startswith("backbone."):
                continue
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        """Get head parameters without weight decay (excludes backbone)."""
        for name, param in self.named_parameters():
            # Skip backbone parameters - they're handled separately
            if name.startswith("backbone."):
                continue
            if "bias" in name or "norm" in name:
                yield (name, param)


class FlexMDMAuxModelShared(Model):
    """
    Auxiliary FlexMDM model with shared backbone.

    The backbone receives training signals from both the main model and this
    auxiliary model via standard backpropagation.
    """

    def __init__(
        self,
        backbone: SharedTransformerBackbone,
        d_model: int,
        dim_feedforward: Optional[int] = None,
        d_cond: Optional[int] = None,
        rate_head_type: Literal["scalar", "discretized"] = "scalar",
        num_rate_bins: int = 1000,
        min_rate: float = 0.01,
        max_rate: float = 100.0,
        rate_head_use_mlp: bool = True,
        inner_autocast: bool = True,
        scalar_fn: Literal["softplus", "exp", "sigmoid"] = "softplus",
        b_um: Optional[float] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_cond = d_cond or d_model // 2
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.inner_autocast = inner_autocast
        self.scalar_fn = scalar_fn
        self.b_um = b_um

        # Rate heads (trainable)
        self.lambda_ins_phi_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )
        self.lambda_unmask_phi_head = ScalarRateHead(
            d_model,
            self.d_cond,
            rate_head_type=rate_head_type,
            num_rate_bins=num_rate_bins,
            min_rate=min_rate,
            max_rate=max_rate,
            use_mlp=rate_head_use_mlp,
            scalar_fn=scalar_fn,
        )

    def forward(
        self,
        z_1: Integer[TT, "batch seq_len"],
        t: Float[TT, "batch"],
        attention_mask: Optional[Bool[TT, "batch seq_len"]] = None,
    ) -> dict:
        """
        Forward pass for the auxiliary model.

        Args:
            z_1: Clean input sequences (B, L)
            t: Time values (B,)
            attention_mask: Attention mask (B, L)

        Returns:
            Dictionary with rate parameters.
        """
        assert attention_mask is not None
        attention_mask = attention_mask.to(torch.bool)

        # Forward through shared backbone (gradients flow normally)
        hidden_states, conditioning = self.backbone(
            z_1,
            t,
            attention_mask,
            inner_autocast=self.inner_autocast,
        )

        # Apply rate heads (trainable)
        b_ins = self.lambda_ins_phi_head(hidden_states, conditioning)
        b_unmask = self.lambda_unmask_phi_head(hidden_states, conditioning)

        if self.b_um is not None:
            b_unmask = torch.full_like(b_unmask, self.b_um)

        return {
            "b_ins": b_ins,
            "b_unmask": b_unmask,
            "a_ins": None,
            "a_unmask": None,
        }

    def get_named_params_for_weight_decay(self):
        """
        Get aux-model parameters that should have weight decay (excludes backbone).

        Note: in the shared-backbone setup, the harness owns the optimizer groups
        for the backbone, so we exclude it here.
        """
        for name, param in self.named_parameters():
            # Skip backbone parameters - handled separately by the harness
            if name.startswith("backbone."):
                continue
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        """
        Get aux-model parameters without weight decay (excludes backbone).

        Note: in the shared-backbone setup, the harness owns the optimizer groups
        for the backbone, so we exclude it here.
        """
        for name, param in self.named_parameters():
            # Skip backbone parameters - handled separately by the harness
            if name.startswith("backbone."):
                continue
            if "bias" in name or "norm" in name:
                yield (name, param)
