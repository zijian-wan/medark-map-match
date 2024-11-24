import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

from sklearn.metrics import cohen_kappa_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layers=1, edge_embed_dim=4):
        super().__init__()
        # Edge class embedding
        self.edge_embedding = nn.Embedding(num_embeddings=20, embedding_dim=edge_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_dim-1+edge_embed_dim, hidden_size, num_layers=n_layers, batch_first=True)

    def forward(self, src, src_lengths):
        # Edge class embedding
        x_edge_class = src[:, :, 3].to(torch.int)
        # Shift edge class by 1 so that 0 denotes padding
        x_edge_class += 1
        x_edge_embedding = self.edge_embedding(x_edge_class)
        x_edge_embedded = torch.cat([src[:, :, :3], x_edge_embedding], dim=2)

        # Pack the sequence
        packed_input = pack_padded_sequence(x_edge_embedded, src_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input)

        return packed_output, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, packed_encoder_output, hidden):
        decoder_output, decoder_hidden = self.lstm(packed_encoder_output, hidden)
        # Unpack
        decoder_output, _ = pad_packed_sequence(decoder_output, batch_first=True)

        # Passing the LSTM output through the fully connected layer
        decoder_output = self.dropout(decoder_output)
        prediction = self.fc(decoder_output)

        return prediction, decoder_hidden


class MatchErrorDetectModel(nn.Module):
    def __init__(
            self, input_dim: int, hidden_size: int, edge_embed_dim: int,
            n_enc_layers: int, n_dec_layers: int, dropout: float = 0.2
    ):
        super().__init__()
        # Trajectory encoder
        self.traj_encoder = LSTMEncoder(
            input_dim, hidden_size, n_layers=n_enc_layers, edge_embed_dim=edge_embed_dim
        )
        # Trajectory decoder
        self.traj_decoder = LSTMDecoder(
            hidden_size, output_dim=1, n_layers=n_dec_layers, dropout=dropout
        )

    def forward(self, src: torch.Tensor, src_lengths: list) -> torch.Tensor:
        packed_encoder_output, hidden = self.traj_encoder(src, src_lengths)
        pred_logits, _ = self.traj_decoder(packed_encoder_output, hidden)
        return pred_logits

    def logits2labels(self, output_logits: torch.Tensor):
        # Convert logits to probabilities for interpretation
        probs = torch.sigmoid(output_logits)
        pred_labels = (probs > 0.5).int()
        return probs, pred_labels

    def predict(self, src: torch.Tensor, src_lengths: list):
        output_logits = self.forward(src, src_lengths)
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

        for x_padded, y_padded, nonpad_mask, padding_mask, x_len, y_len in tqdm(train_loader):
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
            pred_logits = self.forward(src=x_padded, src_lengths=x_len)

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

        for x_padded, y_padded, nonpad_mask, padding_mask, x_len, y_len in tqdm(val_loader):
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
                pred_logits = self.forward(src=x_padded, src_lengths=x_len)

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

