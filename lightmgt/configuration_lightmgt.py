"""LightMGT model configuration."""


class LightMGTConfig:
    """Configuration namespace for LightMGT sub-modules.

    This is a simple configuration container (not a diffusers ConfigMixin).
    The main model class LightMGTTransformer uses ModelMixin + @register_to_config
    for HuggingFace compatibility.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_double_blocks: int = 4,
        num_single_gla_blocks: int = 14,
        num_single_softmax_blocks: int = 6,
        num_attention_heads: int = 16,
        head_dim: int = 64,
        mlp_ratio: float = 2.6875,  # SwiGLU effective ratio: 8/3 ≈ 2.67
        codebook_size: int = 262144,
        mask_token_id: int = 262144,
        vocab_size: int = 262145,
        num_lfq_bits: int = 18,
        gen_head_groups: int = 2,
        gen_head_vocab: int = 512,
        text_hidden_size: int = 1024,  # Qwen3.5-0.8B text hidden_size
        text_max_length: int = 256,
        rope_axes_dim: tuple = (8, 28, 28),
        rope_theta: float = 10000.0,
        label_smoothing: float = 0.1,
        cfg_dropout: float = 0.1,
        use_sandwich_norm: bool = True,
        use_parallel_block: bool = True,
        use_qk_norm: bool = True,
        use_bias: bool = False,
        gla_num_heads: int = None,
        gla_expand_ratio: float = 1.0,
        gla_conv_size: int = 4,
    ):
        self.hidden_size = hidden_size
        self.num_double_blocks = num_double_blocks
        self.num_single_gla_blocks = num_single_gla_blocks
        self.num_single_softmax_blocks = num_single_softmax_blocks
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.mlp_ratio = mlp_ratio
        self.codebook_size = codebook_size
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.num_lfq_bits = num_lfq_bits
        self.gen_head_groups = gen_head_groups
        self.gen_head_vocab = gen_head_vocab
        self.text_hidden_size = text_hidden_size
        self.text_max_length = text_max_length
        self.rope_axes_dim = rope_axes_dim
        self.rope_theta = rope_theta
        self.label_smoothing = label_smoothing
        self.cfg_dropout = cfg_dropout
        self.use_sandwich_norm = use_sandwich_norm
        self.use_parallel_block = use_parallel_block
        self.use_qk_norm = use_qk_norm
        self.use_bias = use_bias
        self.gla_num_heads = gla_num_heads if gla_num_heads is not None else num_attention_heads
        self.gla_expand_ratio = gla_expand_ratio
        self.gla_conv_size = gla_conv_size
