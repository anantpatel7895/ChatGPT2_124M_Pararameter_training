from pydantic import BaseModel, Field

class GPT2Config(BaseModel):
    max_seq_length: int = Field(default=1024, description="max sequence length")
    vocab_size: int = Field(default=50257, description="number of tokens")
    n_xmer_layer: int = Field(default=12, description="number of layers")
    n_head: int = Field(default=12, description="number of heads")
    embd_dim: int = Field(default=768, description="embedding dimension")
    dropout: float = Field(default=0.1, description="dropout probability")

    class Config:
        extra = "forbid"  # optional: disallow unexpected fields