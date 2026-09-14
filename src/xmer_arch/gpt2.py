import torch
import torch.nn as nn

from src.config_models.gpt2_config import GPT2Config
from src.xmer_arch.xmer_layer import XMERLayer


class GPT2(nn.Module):

    """
    REF: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model
    GPT-2 architecture with XMER layers, multi-head self-attention,
    and feedforward MLPs matching HuggingFace implementation.
    """
    def __init__(self, config: GPT2Config):
        super(GPT2, self).__init__()
        
        self.config = config
        
        # Token and position embeddings matching HF naming
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_dim),  # token embeddings
            wpe = nn.Embedding(config.max_seq_length, config.embd_dim),  # position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([XMERLayer(config) for _ in range(config.n_xmer_layer)]),  # transformer blocks
            ln_f = nn.LayerNorm(config.embd_dim),  # final layer norm
        ))
        
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias=False)  # language modeling head

        # Weight tying between input embeddings and output head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/((2 * config.n_xmer_layer)**0.5))

    def _init_weights(self, module):
        """Initialize weights using GPT-2 scheme"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, idx, targets=None):
        """
        Args:
            idx: input token indices, shape (B, T)
            targets: target token indices for loss computation, shape (B, T)
        
        Returns:
            logits: output logits, shape (B, T, vocab_size)
            loss: optional, computed if targets are provided
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_length, f"Cannot forward sequence of length {t}, max is {self.config.max_seq_length}"
        
        # Generate position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # Token embeddings + positional embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, embd_dim)
        pos_emb = self.transformer.wpe(pos)  # (1, T, embd_dim)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_xmer_layer=12, n_head=12, embd_dim=768),  # 124M params
            'gpt2-medium':  dict(n_xmer_layer=24, n_head=16, embd_dim=1024), # 350M params
            'gpt2-large':   dict(n_xmer_layer=36, n_head=20, embd_dim=1280), # 774M params
            'gpt2-xl':      dict(n_xmer_layer=48, n_head=25, embd_dim=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['max_seq_length'] = 1024  # always 1024 for GPT model checkpoints
        config_args['dropout'] = 0.1
        
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # load the pretrained GPT-2 model from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


