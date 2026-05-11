import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Module.Mutation_Module import MutationPolicy, Scoring
from Dataset import PepDataset
from args import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MutationTrainer:
    def __init__(self, func_names, model_paths=None, mode=None):
        self.func_names = func_names
        if model_paths is None:
            model_paths = {f: getattr(args, f'scoring_path_{f}', f'model/prediction/{f}.pth') for f in func_names}
        if mode is None:
            mode = getattr(args, 'scoring_mode', 'regression')
        if isinstance(mode, str):
            mode = {f: mode for f in func_names}

        self.policy_net = MutationPolicy(args.vocab_size, args.emb_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.lr)

        self.scoring_models = {}
        for func in func_names:
            self.scoring_models[func] = Scoring(
                func=func,
                vocab_size=args.vocab_size,
                max_length=args.max_len,
                model_paths=model_paths[func],
                mode=mode[func]
            )

        self.tokenizer = AutoTokenizer.from_pretrained("Data/Vocab", ignore_mismatched_sizes=True)
        self.dataset = PepDataset(path=args.data_path, mode='regression', max_length=args.max_len)
        self.loader = DataLoader(self.dataset, shuffle=True, batch_size=args.batch_size)

        self.policy_save_dir = getattr(args, 'policy_save_path', 'model/policy_net.pth')
        self.change_rate_dir = getattr(args, 'change_rate_path', 'result/change_rate.csv')

    def sample_action(self, input_ids, mask):
        pos_logits, aa_logits = self.policy_net(input_ids)
        pos_probs = F.softmax(pos_logits.masked_fill(mask == 0, -1e9), dim=-1)
        pos = torch.multinomial(pos_probs, 1).squeeze(1)
        aa_logits_at_pos = aa_logits[torch.arange(input_ids.size(0)), pos]
        aa_probs = F.softmax(aa_logits_at_pos, dim=-1)
        new_aa = torch.multinomial(aa_probs, 1).squeeze(1)
        return pos, new_aa, pos_probs, aa_probs

    def mutate_sequence(self, input_ids, pos, new_aa):
        mutated = input_ids.clone()
        for i in range(mutated.size(0)):
            mutated[i, pos[i]] = new_aa[i]
        return mutated

    def _get_weighted_rewards(self, original_ids, mutated_ids, func_names, weights):
        total_reward = 0
        individual_diffs = []
        for i, name in enumerate(func_names):
            scorer = self.scoring_models[name]
            _, orig_norm = scorer(original_ids)
            _, mut_norm = scorer(mutated_ids)
            diff = mut_norm - orig_norm
            total_reward += weights[i] * diff
            individual_diffs.append(diff.detach())
        return total_reward, individual_diffs

    def train(self, epochs, func_names, weights):
        assert len(func_names) == len(weights)
        self.policy_net.train()
        for epoch in range(epochs):
            total_loss = 0
            total_changes = [0.0] * len(func_names)
            total_samples = 0
            for batch in self.loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = (input_ids >= 3).long().to(device)
                pos, new_aa, pos_probs, aa_probs = self.sample_action(input_ids, attention_mask)
                mutated = self.mutate_sequence(input_ids, pos, new_aa)
                rewards, diffs = self._get_weighted_rewards(input_ids, mutated, func_names, weights)
                log_p_pos = torch.log(pos_probs[torch.arange(len(pos)), pos] + 1e-8)
                log_p_aa = torch.log(aa_probs[torch.arange(len(new_aa)), new_aa] + 1e-8)
                log_prob = log_p_pos + log_p_aa
                loss = -(log_prob * rewards).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                for i in range(len(func_names)):
                    total_changes[i] += diffs[i].sum().item()
                total_samples += len(rewards)
            avg_changes = [c / total_samples if total_samples else 0 for c in total_changes]
            self._log_change(epoch, avg_changes, func_names)
            print(f"Epoch {epoch}: Loss {total_loss:.3f}  Changes: {dict(zip(func_names, avg_changes))}")
            if epoch % 10 == 0:
                torch.save(self.policy_net.state_dict(), self.policy_save_dir)

    def _log_change(self, epoch, changes, names):
        os.makedirs(os.path.dirname(self.change_rate_dir), exist_ok=True)
        file_exists = os.path.isfile(self.change_rate_dir)
        with open(self.change_rate_dir, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch'] + [f'Change_{n}' for n in names])
            writer.writerow([epoch] + changes)

    def train_single(self, func_name, epochs=None):
        if epochs is None:
            epochs = args.epochs
        self.train(epochs, [func_name], [1.0])

    def train_multi(self, func_names, weights, epochs=None):
        if epochs is None:
            epochs = args.epochs
        self.train(epochs, func_names, weights)


if __name__ == "__main__":
    trainer = MutationTrainer(
        func_names=['E_coli', 'Hemolysis'],
        model_paths={'E_coli': r'D:\2026\PepMutation\Data\Result\Ecoli_reg\model_best.pth',
                     'Hemolysis': r'D:\2026\PepMutation\Data\Result\Ecoli_reg\model_best.pth'},

        mode='regression'
    )
    trainer.train_multi(func_names=['E_coli', 'Hemolysis'], weights=[0.5, 0.5]
        )