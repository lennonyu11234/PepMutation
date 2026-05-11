import torch
import torch.nn as nn
from Module.Score_module import GRURegressionModel, GRUClassificationModel
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MutationPolicy(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.pos_head = nn.Linear(hidden_dim * 2, 1)
        self.aa_head = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.encoder(x)
        pos_logits = self.pos_head(out).squeeze(-1)  # [B, L]
        aa_logits = self.aa_head(out)                # [B, L, V]
        return pos_logits, aa_logits


class Scoring(nn.Module):
    def __init__(self, func, vocab_size, max_length, model_paths, mode='regression'):
        super().__init__()
        self.func = func
        self.mode = mode
        if self.mode == 'regression':
            self.scoring_model = GRURegressionModel(vocab_size, max_length).to(device)
            checkpoint = torch.load(model_paths, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.scoring_model.load_state_dict(state_dict)
        elif self.mode == 'classification':
            self.scoring_model = GRUClassificationModel(vocab_size, max_length).to(device)
            checkpoint = torch.load(model_paths, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            self.scoring_model.load_state_dict(state_dict)
        self.scoring_model.eval()

        self.norm_params = {
            'Assemble': (0.99, 2.62),
            'E_coli': (3.2, 6.0),
            'Hemolysis': (3.0, 6.72),
        }

    def forward(self, input_ids):
        with torch.no_grad():
            output = self.scoring_model(input_ids.to(device))

        if self.mode == 'regression':
            if self.func == 'Assemble':
                min_val, max_val = self.norm_params[self.func]
                normalized = 2 * ((output - min_val) / (max_val - min_val)) - 1
                return output, normalized
            elif self.func == 'E_coli':
                mic = torch.pow(10, -output) * 1e6
                min_val, max_val = self.norm_params[self.func]
                normalized = 2 * ((output - min_val) / (max_val - min_val)) - 1
                return mic, normalized
            elif self.func == 'Hemolysis':
                mic = torch.pow(10, -output) * 1e6
                min_val, max_val = self.norm_params[self.func]
                normalized = 2 * ((output - min_val) / (max_val - min_val)) - 1
                return mic, -normalized
            else:
                raise ValueError(f"Unknown func: {self.func}")

        elif self.mode == 'classification':
            pred_class = torch.argmax(output, dim=1)  # 0 或 1
            reward = torch.where(pred_class == 1,
                                 torch.tensor(0.5, device=output.device),
                                 torch.tensor(-0.5, device=output.device))
            return pred_class.float(), reward
        else:
            raise ValueError(f"Unknown mode: {self.mode}")