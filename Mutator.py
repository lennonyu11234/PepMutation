import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from Module.Mutation_Module import MutationPolicy
from args import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Mutator:
    def __init__(self, policy_path, vocab_size=None, hidden_dim=None,
                 max_length=None, tokenizer_path="Data/Vocab"):
        if vocab_size is None:
            vocab_size = args.vocab_size
        if hidden_dim is None:
            hidden_dim = args.emb_dim
        if max_length is None:
            max_length = args.max_len

        self.max_length = max_length
        self.vocab_size = vocab_size

        self.policy_net = MutationPolicy(vocab_size, hidden_dim).to(device)
        checkpoint = torch.load(policy_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, ignore_mismatched_sizes=True)

    def _sequence_to_input_ids(self, sequence):
        seq = sequence.upper().strip()
        tokens = [self.tokenizer.bos_token] + list(seq) + [self.tokenizer.eos_token]
        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.pad_token] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_length]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, attention_mask

    def _input_ids_to_sequence(self, input_ids):

        ids = input_ids.squeeze(0).cpu()
        seq = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        seq = ''.join([c for c in seq if c.isalpha()])
        return seq

    def sample_action(self, input_ids, attention_mask):
        pos_logits, aa_logits = self.policy_net(input_ids)
        pos_probs = F.softmax(pos_logits.masked_fill(attention_mask == 0, -1e9), dim=-1)
        pos = torch.multinomial(pos_probs, 1).squeeze(1)
        aa_logits_at_pos = aa_logits[0, pos[0]]
        aa_probs = F.softmax(aa_logits_at_pos, dim=-1)
        new_aa = torch.multinomial(aa_probs, 1)
        return pos, new_aa

    def mutate(self, sequence, num_steps=1):
        input_ids, attention_mask = self._sequence_to_input_ids(sequence)
        current_ids = input_ids.clone()
        mutated_seqs = [sequence]

        for _ in range(num_steps):
            pos, new_aa = self.sample_action(current_ids, attention_mask)
            current_ids[0, pos[0]] = new_aa
            mutated_seq = self._input_ids_to_sequence(current_ids)
            mutated_seqs.append(mutated_seq)

        return mutated_seqs

    def mutate_single_step(self, sequence):
        result = self.mutate(sequence, num_steps=1)
        return result[0], result[1]


# ---------- 使用示例 ----------
if __name__ == "__main__":
    mutator = Mutator(policy_path=args.policy_save_path)
    test_peptide = "GLFDVIKKVAGLGL"
    orig, mut = mutator.mutate_single_step(test_peptide)
    print(f"Original : {orig}")
    print(f"Mutated  : {mut}")
    trajectory = mutator.mutate(test_peptide, num_steps=5)
    for i, seq in enumerate(trajectory):
        print(f"Step {i}: {seq}")