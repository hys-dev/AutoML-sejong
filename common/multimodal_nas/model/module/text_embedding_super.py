import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class TextembedSuper(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, args, vocab_size=30522, hidden_size=768, pad_token_id=0, max_position_embeddings=512,
                 type_vocab_size=2, layer_norm_eps=1e-12, hidden_dropout_prob=0.1, position_embedding_type="absolute"):
        super(TextembedSuper, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.position_embedding_type = position_embedding_type
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # sampled config
        self.sample_embed_dim = None
        self.we_weight = None
        self.pe_weight = None
        self.te_weight = None
        self.ln_weight = None
        self.ln_bias = None
        
        if "supernet" in args.resume:
            self.load_pretrained(embed_dim=hidden_size)

    @torch.no_grad()
    def load_pretrained(self, embed_dim, ckpt: str = "bert-base-uncased"):
        from transformers import BertModel
        bert = BertModel.from_pretrained(ckpt)

        self.word_embeddings.weight.copy_(bert.embeddings.word_embeddings.weight[:, :embed_dim])
        self.position_embeddings.weight.copy_(bert.embeddings.position_embeddings.weight[:, :embed_dim])
        self.token_type_embeddings.weight.copy_(bert.embeddings.token_type_embeddings.weight[:, :embed_dim])
        self.LayerNorm.weight.copy_(bert.embeddings.LayerNorm.weight[:embed_dim])
        self.LayerNorm.bias.copy_(bert.embeddings.LayerNorm.bias[:embed_dim])
        del bert

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.we_weight = self.word_embeddings.weight[:, :self.sample_embed_dim]
        self.pe_weight = self.position_embeddings.weight[:, :self.sample_embed_dim]
        self.te_weight = self.token_type_embeddings.weight[:, :self.sample_embed_dim]
        self.ln_weight = self.LayerNorm.weight[:self.sample_embed_dim]
        self.ln_bias = self.LayerNorm.bias[:self.sample_embed_dim]

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            if self.sample_embed_dim is not None:
                inputs_embeds = F.embedding(input_ids, self.we_weight, self.word_embeddings.padding_idx)
            else:
                inputs_embeds = self.word_embeddings(input_ids)

        if self.sample_embed_dim is not None:
            token_type_embeddings = F.embedding(token_type_ids, self.te_weight)
        else:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            if self.sample_embed_dim is not None:
                position_embeddings = F.embedding(position_ids, self.pe_weight)
            else:
                position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if self.sample_embed_dim is not None:
            embeddings = F.layer_norm(embeddings, (self.sample_embed_dim,), self.ln_weight, self.ln_bias,
                                      self.LayerNorm.eps)
        else:
            embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings

    def calc_sampled_param_num(self):
        if self.sample_embed_dim is None:
            return sum(p.numel() for p in self.parameters())

        total_params = 0
        total_params += self.we_weight.numel()
        total_params += self.pe_weight.numel()
        total_params += self.te_weight.numel()
        total_params += self.ln_weight.numel()
        total_params += self.ln_bias.numel()
        return total_params

    def get_complexity(self, sequence_length):
        embed_dim = self.sample_embed_dim if self.sample_embed_dim is not None else self.LayerNorm.normalized_shape[0]

        total_flops = 0
        total_flops += sequence_length * embed_dim * 3
        total_flops += sequence_length * embed_dim * 5
        return total_flops