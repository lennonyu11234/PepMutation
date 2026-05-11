import math
import json
import os
import logging
from datetime import datetime

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Dataset import PepDataset
from args import args
from Module.Score_module import (
    CNNRegressionModel,
    LSTMRegressionModel,
    GRURegressionModel,
    TransformerRegressionModel,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(99)


class RegTrainer:
    def __init__(
        self,
        data_path="Data/Dataset/E_coli.csv",
        mode="regression",
        model_type="transformer",
        batch_size=32,
        lr=1e-3,
        epochs=50,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        output_dir="outputs",
        experiment_name=None,
        seed=99,
    ):
        self.mode = mode
        self.model_type = model_type
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logging()
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Device: {device}")

        self._save_config()
        self._setup_data(data_path)
        self._setup_model()
        self._setup_optimizer()

        self.metrics_df = pd.DataFrame()
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _setup_logging(self):
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        fh = logging.FileHandler(os.path.join(self.output_dir, "training.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _save_config(self):
        config = {
            "mode": self.mode,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Config saved to {config_path}")

    def _setup_data(self, data_path):
        full_dataset = PepDataset(path=data_path, mode=self.mode, max_length=args.max_len)
        self.vocab_size = args.vocab_size
        self.max_length = args.max_len

        total_len = len(full_dataset)
        train_len = int(self.train_ratio * total_len)
        val_len = int(self.val_ratio * total_len)
        test_len = total_len - train_len - val_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.logger.info(
            f"Data split: {train_len} train, {val_len} val, {test_len} test"
        )

    def _setup_model(self):
        if self.model_type == "cnn":
            self.model = CNNRegressionModel(self.vocab_size, self.max_length).to(device)
        elif self.model_type == "lstm":
            self.model = LSTMRegressionModel(self.vocab_size, self.max_length).to(device)
        elif self.model_type == "gru":
            self.model = GRURegressionModel(self.vocab_size, self.max_length).to(device)
        elif self.model_type == "transformer":
            self.model = TransformerRegressionModel(self.vocab_size, self.max_length).to(device)
        else:
            raise ValueError(f"Unsupported model_type for regression: {self.model_type}")
        self.logger.info(f"Model: {self.model_type} regression")

    def _setup_optimizer(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if is_best:
            path = os.path.join(self.output_dir, "model_best.pth")
        else:
            path = os.path.join(self.output_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, path)
        if is_best:
            self.logger.info(f"Saved best model to {path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].float().to(device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(
            f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {avg_loss:.4f}"
        )
        return avg_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].float().to(device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)

        rmse = math.sqrt(mean_squared_error(labels_np, preds_np))
        mae = mean_absolute_error(labels_np, preds_np)
        r2 = r2_score(labels_np, preds_np)
        pearson_corr, _ = pearsonr(labels_np, preds_np)

        metrics_row = {
            "Epoch": epoch + 1,
            "Val Loss": avg_loss,
            "Val RMSE": rmse,
            "Val MAE": mae,
            "Val R2": r2,
            "Val Pearson": pearson_corr,
        }
        self.metrics_df = pd.concat(
            [self.metrics_df, pd.DataFrame([metrics_row])], ignore_index=True
        )

        self.logger.info(
            f"Epoch {epoch + 1}/{self.epochs} | Val Loss: {avg_loss:.4f} | "
            f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Pearson: {pearson_corr:.4f}"
        )

        return avg_loss, rmse, mae, r2, pearson_corr

    def evaluate(self, data_loader, label="Evaluation"):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].float().to(device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        labels_np = np.array(all_labels)
        preds_np = np.array(all_predictions)

        rmse = math.sqrt(mean_squared_error(labels_np, preds_np))
        mae = mean_absolute_error(labels_np, preds_np)
        r2 = r2_score(labels_np, preds_np)
        pearson_corr, _ = pearsonr(labels_np, preds_np)

        metrics = {
            "Loss": avg_loss,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson": pearson_corr,
        }

        self.logger.info(
            f"{label} | Loss: {avg_loss:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Pearson: {pearson_corr:.4f}"
        )
        return metrics

    def test(self):
        test_metrics = self.evaluate(self.test_loader, label="Test")
        test_path = os.path.join(self.output_dir, "test_metrics.json")
        with open(test_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        self.logger.info(f"Test metrics saved to {test_path}")
        return test_metrics

    def train(self):
        self.logger.info("Start training...")
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            val_loss, _, _, _, _ = self.val_epoch(epoch)
            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)

        final_path = os.path.join(self.output_dir, "model_final.pth")
        torch.save(self.model.state_dict(), final_path)
        self.logger.info(f"Final model saved to {final_path}")

        metrics_path = os.path.join(self.output_dir, "training_metrics.csv")
        self.metrics_df.to_csv(metrics_path, index=False)
        self.logger.info(f"Training metrics saved to {metrics_path}")

        self.logger.info("Evaluating on test set...")
        self.test()
        self.logger.info("Training completed.")


if __name__ == "__main__":
    trainer = RegTrainer(
        data_path="../Data/Dataset/E_coli.csv",
        mode="regression",
        model_type="gru",
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir="../Data/Result",
        experiment_name="Ecoli_reg",
        seed=99,
    )
    trainer.train()