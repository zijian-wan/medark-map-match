import os
import math
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CosinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        Positional Encoding is used to inject the position information of each token in the input sequence.
        It uses sine and cosine functions of different frequencies to generate the positional encoding.

        Param:
        ---
        d_model: int
            Dimension of the model's embeddings
        dropout: float
            Dropout rate
        max_len: int
            Maximum length of the sequence for which positional encodings are pre-computed
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Position indices for each position in the sequence
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Scale the position indices in a specific way
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # Positional encodings
        pe = torch.zeros(max_len, d_model)
        # The sine function is applied to the even indices and the cosine function to the odd indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe is part of the module's state but will not be considered a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape: (batch_size, seq_len, d_model)
        """
        pos_embeddings =  self.pe[:, :x.size(1)]
        pos_embeddings = self.dropout(pos_embeddings)
        # pos_embeddings shape: (seq_len, d_model)
        # When doing "x + pos_embeddings", PyTorch automatically broadcasts pos_embeddings to match
        # the batch_size dimension of x
        return pos_embeddings


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Positional Encoding is used to inject the position information of each token in the input sequence.
        It uses sine and cosine functions of different frequencies to generate the positional encoding.

        Param:
        ---
        d_model: int
            Dimension of the model's embeddings
        dropout: float
            Dropout rate
        """
        super().__init__()
        self.fc_layer = nn.Linear(1, d_model)  # Position
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape: (batch_size, seq_len, d_model)
        """
        # Batch size and sequence length
        batch_size, seq_len = x.shape[:2]
        # Create a position tensor of shape (batch_size, seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(0).expand(batch_size, -1).to(x.device)
        # Add a new dimension to get shape (batch_size, seq_len, 1)
        position = position.unsqueeze(-1)
        pos_embeddings =  self.fc_layer(position)
        pos_embeddings = self.dropout(pos_embeddings)
        return pos_embeddings


class TrajectoryEncoder(nn.Module):
    def __init__(
            self, input_dim: int, d_model: int, n_head: int, d_hid: int, edge_embed_dim: int,
            n_enc_layers: int, dropout: float = 0.2, temporal_emb_type: str = "cosine"
    ):
        super().__init__()
        self.d_model = d_model
        self.trans_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_hid, dropout=dropout, batch_first=True
        )
        self.trans_encoder = nn.TransformerEncoder(self.trans_enc_layer, num_layers=n_enc_layers)
        # Spatial embedding
        self.spatial_embedding = nn.Linear(input_dim-1+edge_embed_dim, d_model)
        # Temporal embedding
        self.temporal_emb_type = temporal_emb_type
        if temporal_emb_type == "cosine":
            self.temporal_embedding = CosinePositionalEncoding(d_model, dropout)
        elif temporal_emb_type == "learned":
            self.temporal_embedding = LearnedPositionalEncoding(d_model, dropout)
        # Edge class embedding
        self.edge_embedding = nn.Embedding(num_embeddings=20, embedding_dim=edge_embed_dim, padding_idx=0)

    def init_weights(self) -> None:
        init_range = 0.1
        self.input_projection.weight.data.uniform_(-init_range, init_range)

    def forward(self, src: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Input:
        ---
            src: nn.Tensor, shape [batch_size, seq_len, input_dim]
            key_padding_mask: nn.Tensor, shape [batch_size, seq_len]
                provides specified elements in the key to be ignored by the attention
                If a BoolTensor is provided, the positions with the value of True will be ignored.

        Output:
        ---
            output: nn.Tensor, shape [batch_size, seq_len, 1]
        """
        # Edge class embedding
        x_edge_class = src[:, :, 3].to(torch.int)
        # Shift edge class by 1 so that 0 denotes padding
        x_edge_class += 1
        x_edge_embedding = self.edge_embedding(x_edge_class)
        x_edge_embedded = torch.cat([src[:, :, :3], x_edge_embedding], dim=2)

        # Spatial embedding
        src_spatial = self.spatial_embedding(x_edge_embedded) * math.sqrt(self.d_model)  # Normalization
        # Temporal embedding
        temp_embed = self.temporal_embedding(src_spatial)  # take the spatially embedded source sequence as the input
        src_embed = src_spatial + temp_embed

        # Transformer
        encoder_out = self.trans_encoder(src_embed, src_key_padding_mask=key_padding_mask)

        return encoder_out


class TrajectoryDecoder(nn.Module):
    def __init__(
            self, d_model: int, n_head: int, d_hid: int, n_dec_layers: int, dropout: float = 0.2
    ):
        super().__init__()
        # Decoder embedding and positional encoding
        self.dec_embedding = nn.Linear(1, d_model)  # Map match error: 1d
        self.pos_encoding = CosinePositionalEncoding(d_model, dropout)
        # Transformer decoder layer
        self.trans_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_hid, dropout=dropout, batch_first=True
        )
        # Transformer decoder
        self.trans_decoder = nn.TransformerDecoder(self.trans_dec_layer, num_layers=n_dec_layers)
        # Output projection layer for binary classification
        self.output_projection = nn.Linear(d_model, 1)

    def init_weights(self) -> None:
        init_range = 0.1
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-init_range, init_range)

    def forward(
            self, encoder_out: torch.Tensor, key_padding_mask: torch.Tensor, y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Input:
        ---
            encoder_out: nn.Tensor, shape [batch_size, seq_len, d_model]
            key_padding_mask: nn.Tensor, shape [batch_size, seq_len]
                provides specified elements in the key to be ignored by the attention
                If a BoolTensor is provided, the positions with the value of True will be ignored.

        Output:
        ---
            output: nn.Tensor, shape [batch_size, seq_len, 1]
        """
        # Batch size and sequence length
        batch_size, seq_len = key_padding_mask.shape
        # Start token, marks the start of the target sequence
        start_token = -2
        start_tokens_ts = torch.full((batch_size, 1, 1), start_token, dtype=y.dtype).to(encoder_out.device)

        if y is not None:  # During training, target sequences exist
            # (Shifted) target sequence
            # Concatenate the start tokens to 'y', shifting 'y' to the right by 1 position
            # Truncate the last element of 'y' to keep the sequence length constant
            tgt = torch.cat((start_tokens_ts, y[:, :-1, :]), dim=1)
            # tgt shape: (batch_size, seq_len, 1)

            # Target mask to prevent the decoder to attend to future tokens
            # key_padding_mask (boolean mask) and tgt_mask have to match in types
            tgt_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=encoder_out.device), diagonal=1
            )
            # No batch dimension is needed for tgt_mask

            # Target embedding
            tgt_embed = self.dec_embedding(tgt)
            # tgt_embed shape: (batch_size, seq_len, d_model)
            tgt_embed += self.pos_encoding(tgt_embed)

            # Transformer decoder output
            decoder_out = self.trans_decoder(
                tgt_embed, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask,
                memory=encoder_out, memory_key_padding_mask=key_padding_mask
            )

            # Output projection
            output = self.output_projection(decoder_out)

        else:  # During inference, no complete target sequences
            # Input and output in a loop
            output = torch.zeros(batch_size, seq_len, 1, device=encoder_out.device)
            tgt = torch.full((batch_size, 1), -2, dtype=torch.float).to(encoder_out.device)
            for i in range(seq_len):
                tgt_embed = self.dec_embedding(tgt)
                tgt_embed += self.pos_encoding(tgt_embed)
                decoder_out = self.trans_decoder(tgt_embed, memory=encoder_out, memory_key_padding_mask=key_padding_mask)
                step_output = self.output_projection(decoder_out[:, -1:, :])  # (batch_size, 1, d_model)
                output[:, i:i+1, :] = step_output  # "i:i+1" retains the original number of dimensions
                tgt = step_output.detach()

        return output


class MatchErrorDetectModel(nn.Module):
    def __init__(
            self, input_dim: int, d_model: int, n_head: int, d_hid: int, edge_embed_dim: int,
            n_enc_layers: int, n_dec_layers: int, dropout: float = 0.2
    ):
        """
        Param:
        ---
        input_dim: int
            Dimension of the model's input
        d_model: int
            Dimension of the model's embeddings
        n_head: int
            Number of heads in multi-head attention
        d_hid: int
            Dimension of the feedforward network
        n_enc_layers: int
            Number of sub-encoder-layers in the encoder
        n_dec_layers: int
            Number of sub-decoder-layers in the decoder
        dropout: float
            Dropout rate
        """
        super().__init__()
        # Trajectory encoder
        self.traj_encoder = TrajectoryEncoder(
            input_dim=input_dim, d_model=d_model, n_head=n_head, d_hid=d_hid, edge_embed_dim=edge_embed_dim,
            n_enc_layers=n_enc_layers, dropout=dropout, temporal_emb_type="cosine"
        )
        # Trajectory decoder
        self.traj_decoder = TrajectoryDecoder(
            d_model=d_model, n_head=n_head, d_hid=d_hid, n_dec_layers=n_dec_layers, dropout=dropout
        )

    def forward(
            self, src: torch.Tensor, key_padding_mask: torch.Tensor = None, gt: torch.Tensor = None
    ) -> torch.Tensor:
        encoder_out = self.traj_encoder(src, key_padding_mask)
        pred_logits = self.traj_decoder(encoder_out, key_padding_mask, gt)
        return pred_logits

    def logits2labels(self, output_logits: torch.Tensor):
        # Convert logits to probabilities for interpretation
        probs = torch.sigmoid(output_logits)
        pred_labels = (probs > 0.5).int()
        return probs, pred_labels

    def predict(
            self, src: torch.Tensor, key_padding_mask: torch.Tensor = None, gt: torch.Tensor = None
    ):
        output_logits = self.forward(src, key_padding_mask, gt)
        probs, pred_labels = self.logits2labels(output_logits)
        return probs, pred_labels

    def train_model(
            self, train_loader, epoch, train_batch_size, optimizer, criterion
    ):
        # Training
        self.traj_encoder.train()
        self.traj_decoder.train()

        # Loss and evaluation metrics
        loss_batchsum = 0
        kappa_batchsum = 0
        acc_batchsum = 0
        batch_skip = 0

        for x_padded, y_padded, nonpad_mask, padding_mask in tqdm(train_loader):
            if x_padded.shape[0] < train_batch_size:  # not enough data
                batch_skip += 1  # skip the last batch
                continue

            # Data sent to gpu
            x_padded = x_padded.to(device)
            # Unsqueeze y to give it the 3rd dimension
            y_padded = y_padded.unsqueeze(-1).to(device)
            nonpad_mask = nonpad_mask.to(device)
            padding_mask = padding_mask.to(device)

            # Zero gradients of parameters
            optimizer.zero_grad()

            # Predict the logits (not the labels) for training
            pred_logits = self.forward(
                src=x_padded, key_padding_mask=padding_mask, gt=y_padded
            )

            # Reshape outputs and labels to be compatible with BCEWithLogitsLoss
            pred_logits_flat = pred_logits.flatten()  # Flatten the output
            y_padded_flat = y_padded.flatten()  # Flatten the labels
            nonpad_mask_flat = nonpad_mask.flatten()  # Flatten the mask

            # Output and labels without padding (but flattened)
            y_gt = torch.masked_select(y_padded_flat, nonpad_mask_flat)
            y_logits = torch.masked_select(pred_logits_flat, nonpad_mask_flat)

            # Loss
            bce_loss = criterion(y_logits, y_gt)

            # Backward pass and optimization
            bce_loss.backward()
            optimizer.step()

            # Calculate error
            loss_batchsum += bce_loss.item()
            with torch.no_grad():
                _, y_pred = self.logits2labels(output_logits=y_logits)
                kappa, accuracy = calc_eval_metrics(y_pred, y_gt)
                kappa_batchsum += kappa
                acc_batchsum += accuracy

        # Batch average
        with torch.no_grad():
            loss_batchavg = loss_batchsum / (len(train_loader) - batch_skip)
            kappa_batchavg = kappa_batchsum / (len(train_loader) - batch_skip)
            acc_batchavg = acc_batchsum / (len(train_loader) - batch_skip)

        # Progress bar
        print(
            f"Epoch {epoch+1} | Training | train_loss {loss_batchavg:.4f} | train_kappa {kappa_batchavg:.4f}" +
            f" | train_acc {acc_batchavg:.4f}"
        )

        return loss_batchavg, kappa_batchavg

    def eval_model(self, val_loader, eval_batch_size, criterion):
        # Validating
        self.traj_encoder.eval()
        self.traj_decoder.eval()

        # Loss and evaluation metrics
        loss_batchsum = 0
        kappa_batchsum = 0
        acc_batchsum = 0
        batch_skip = 0

        for x_padded, y_padded, nonpad_mask, padding_mask in tqdm(val_loader):
            if x_padded.shape[0] < eval_batch_size:  # not enough data
                batch_skip += 1  # skip the last batch
                continue

            # Data sent to gpu
            x_padded = x_padded.to(device)
            # Unsqueeze y to give it the 3rd dimension
            y_padded = y_padded.unsqueeze(-1).to(device)
            nonpad_mask = nonpad_mask.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                # Predict the logits (not the labels) for evaluation
                pred_logits = self.forward(
                    src=x_padded, key_padding_mask=padding_mask, gt=y_padded
                )

                # Reshape outputs and labels to be compatible with BCEWithLogitsLoss
                pred_logits_flat = pred_logits.flatten()  # Flatten the output
                y_padded_flat = y_padded.flatten()  # Flatten the labels
                nonpad_mask_flat = nonpad_mask.flatten()  # Flatten the mask

                # Output and labels without padding (but flattened)
                y_gt = torch.masked_select(y_padded_flat, nonpad_mask_flat)
                y_logits = torch.masked_select(pred_logits_flat, nonpad_mask_flat)

                # Loss
                bce_loss = criterion(y_logits, y_gt)

                # Calculate error
                loss_batchsum += bce_loss.item()
                _, y_pred = self.logits2labels(output_logits=y_logits)
                kappa, accuracy = calc_eval_metrics(y_pred, y_gt)
                kappa_batchsum += kappa
                acc_batchsum += accuracy

        # Batch average
        with torch.no_grad():
            loss_batchavg = loss_batchsum / (len(val_loader) - batch_skip)
            kappa_batchavg = kappa_batchsum / (len(val_loader) - batch_skip)
            acc_batchavg = acc_batchsum / (len(val_loader) - batch_skip)

        # Progress bar
        print(
            f"| Validating | val_loss {loss_batchavg:.4f} | val_kappa {kappa_batchavg:.4f}" +
            f" | val_acc {acc_batchavg:.4f}"
        )

        return loss_batchavg, kappa_batchavg



