<h1 align="center">AI-Guided Multi-Objective Optimization of Jelleine I Yields a Gram-Negative Antimicrobial Peptide Lead with Hydrogel-Forming Capacity</h1>

---

<p align="center">
  <img src="Data/Figure1.png" width="900">
</p>
<p align="center">
  <em>Figure 1. Multi-objective deep learning and reinforcement learning framework for peptide property prediction and goal-directed optimization.</em>
</p>

---

## Abstract

Antimicrobial peptides (AMPs) are promising alternatives to antibiotics, but their development is often limited by the difficulty of balancing antibacterial potency, biocompatibility, and formulation-related properties. Here, using Jelleine I (J1) as a scaffold, we developed a multi-task learning and reinforcement learning-based framework for directed peptide optimization across antibacterial activity, hemolytic liability, and self-assembly propensity. Experimental screening identified YB18 as a lead sequence with balanced activity against Gram-negative reference strains and clinically derived resistant isolates, acceptable in vitro biocompatibility, and rapid membrane-associated bactericidal behavior. In an *E. coli*-infected wound model, YB18 reduced bacterial burden in vivo. Importantly, YB18 retained the self-assembly behavior of the parent scaffold and formed a viscoelastic hydrogel under buffered conditions. The YB18 hydrogel further showed efficacy in infected wounds and favorable local tolerance after repeated topical administration, supporting YB18 as a promising peptide-based material for local anti-infective applications.

## Highlights

- Multi-objective peptide optimization based on **multi-task learning** and **reinforcement learning**
- Optimization targets include:
  - Antibacterial activity
  - Hemolytic liability
  - Self-assembly propensity
- Identification of **YB18** as a Gram-negative antimicrobial peptide lead
- Experimental validation against reference strains and clinically derived resistant isolates
- Demonstration of hydrogel-forming capacity and topical anti-infective potential

---

## Key Modules

- **Score_module** – Contains neural network architectures for predicting peptide properties.
- **Scoring/** – Fully equipped training loops for regression and classification tasks, with logging, checkpointing, and evaluation.
- **Mutation_Module** – Defines the policy network that proposes which amino acid to change and where, and the `Scoring` module that loads your trained predictors as reward functions.
- **Mutation_trainer** – Uses policy gradient (REINFORCE) to train the mutation policy to maximise one or multiple reward signals.
- **Mutation** – Loads a trained policy and applies it to a single peptide sequence to produce mutants.

---

## ⚙️ Environment & Dependencies
```bash
- Python ≥ 3.8
- PyTorch ≥ 1.10
- Transformers (HuggingFace)
- pandas, numpy, scikit‑learn, scipy, rdkit (for fingerprint visualisations, optional)
```
## 🚀 Usage

### 1. Train Scoring Models

Prepare a CSV with columns `SEQUENCE` and target (e.g. `MIC`).

**Regression (e.g. MIC prediction)**

```bash
python Reg_trainer.py \
    --data_path Data/Dataset/E_coli.csv \
    --model_type gru \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-3 \
    --output_dir experiments/Ecoli
```
### 2. Train Mutation Policy
```bash
from Mutation_trainer import MutationTrainer

trainer = MutationTrainer(
    func_names=['E_coli', 'Hemolysis'],
    model_paths={
        'E_coli': 'experiments/Ecoli/model_best.pth',
        'Hemolysis': 'experiments/Hemo/model_best.pth'
    },
    mode='regression'   # or a dict like {'E_coli':'regression', 'Hemolysis':'classification'}
)

# Single objective
trainer.train_single('E_coli', epochs=200)

# Multi-objective with weights
trainer.train_multi(['E_coli', 'Hemolysis'], weights=[0.6, 0.4], epochs=200)

# Three objectives
trainer.train_multi(['Assemble', 'E_coli', 'Hemolysis'], weights=[0.3, 0.4, 0.3])
```

### 3. Mutate a Peptide (Inference)
```bash
from Mutator import Mutator

mutator = Mutator(policy_path="model/policy_net.pth")

peptide = "PFKLSLHL"
orig, mutant = mutator.mutate_single_step(peptide)
print(f"{orig} → {mutant}")

# Multi-step trajectory
for i, seq in enumerate(mutator.mutate(peptide, num_steps=3)):
    print(f"Step {i}: {seq}")
```

```test
D:\Anaconda\envs\archeologist\python.exe D:\2026\PepMutation\Mutator.py 
Original : PFKLSLHL
Mutated  : PFKLSLWL
Step 0: RFKLSLRL
Step 1: RFKLSLRL
Step 2: RFKLILRL
Step 3: RFRLILRL
Step 4: RFRLMLRL

Process finished with exit code 0

```

