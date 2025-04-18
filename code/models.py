import numpy as np
import torch
import torch.nn as nn

################################################################################
######################## Base DistilBERT Model #################################
################################################################################

class DistilBertConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class DistilBertEmbeddings(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        w_emb = self.word_embeddings(input_ids)
        p_emb = self.position_embeddings(pos_ids)
        return self.dropout(w_emb + p_emb)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def compute_attention(self, Q, K, V, mask=None):
        """
        Returns single-headed attention between Q, K, and V.
        """
        scores = Q @ K.transpose(-1, -2) / np.sqrt(self.d_k)  # shape: (B, h, L, L)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights @ V

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        attention = self.compute_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention))
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc2(self.ReLU(self.fc1(x)))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask=None):

        x = self.norm1(x + self.dropout(self.self_attn(x, mask)))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DistilBertModel(nn.Module):
    '''
    Encoder-only BERT model.
    '''
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.embeddings = DistilBertEmbeddings(config)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.hidden_size,
                num_heads=config.num_attention_heads,
                d_ff=config.intermediate_size,
                p=config.hidden_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, mask=None):
        x = self.embeddings(input_ids)
        all_hidden_states = []

        for layer in self.encoder_layers:
            x = layer(x, mask)
            all_hidden_states.append(x)

        return x, all_hidden_states

################################################################################
################ DistilBERT Model For Masked Language Modeling #################
################################################################################

class DistilBertForMaskedLM(nn.Module):
    '''
    Encoder-only BERT model with MLM head.
    '''
    def __init__(self, config: DistilBertConfig):
        super().__init__() 
        self.distilbert = DistilBertModel(config)

        # MLM Head
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Share weights as mentioned in paper
        self.fc2.weight = self.distilbert.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Attention
        hidden_states, all_hidden_states = self.distilbert(input_ids, attention_mask)


        # MLM Head
        x = self.activation(self.fc1(hidden_states))
        x = self.dropout(x)
        logits = self.fc2(x)

        # When training, return cross entropy loss between actual vocab token 
        # and predicted vocab token.
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # Ignore non-masked tokens
            # logits: (B, SEQ_LEN, VOCAB_SIZE) -> (B * SEQ_LEN, VOCAB SIZE) -> Vocab token prediction per non-masked sequence position
            # labels: (B, SEQ_LEN) -> (B * SEQ_LEN) -> True vocab token label per non-masked sequence position
            logits_flat = torch.flatten(logits, start_dim=0, end_dim=1)
            labels_flat = labels_flat = labels.flatten()
            loss = loss_fn(logits_flat, labels_flat) 

        return {
            "loss": loss, # For training
            "logits": logits, # For inference / KL-Div loss training
            "hidden_states": all_hidden_states # For Cosine embedding loss training
        }
