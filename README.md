# Evaluating Sparse Autoencoder Faithfulness Under Causal Interventions

An independent mechanistic interpretability project investigating whether sparse autoencoder (SAE) features capture behaviorally relevant mechanisms in transformer language models.

## Research Question

How much of a model's behavior can be recovered by intervening on selected SAE features, and do the features with the strongest intervention effects also have measurable effects when ablated?

The project examines the distinction between representing a model's activations and identifying features that causally influence its behavior.

## Overview

Sparse autoencoders decompose neural network activations into feature representations that can be studied individually. However, good activation reconstruction alone does not establish that individual features faithfully represent the mechanisms responsible for a model's behavior.

This project uses GPT-2 Small and an SAE from SAE-Lens to investigate feature-level interventions on an indirect-object identification (IOI)-style task. It compares clean and corrupted prompts, measures logit-difference recovery following latent interventions, evaluates combinations of candidate features, and tests individual features through ablation.

## Methods

* **Model:** GPT-2 Small
* **Interpretability tools:** TransformerLens and SAE-Lens
* **SAE release:** `gpt2-small-res-jb`
* **SAE hook point:** `blocks.8.hook_resid_pre`
* **Task:** An IOI-style prompt pair involving John, Mary, and a book
* **Evaluation metric:** Logit difference between the clean and corrupted indirect-object token candidates

### Experiments

1. **Individual latent intervention:** Rank active SAE features according to their effect on recovery of the clean–corrupted logit-difference gap.
2. **Combined latent intervention:** Test whether progressively combining top-ranked features increases recovery.
3. **Latent ablation:** Remove selected latent contributions and measure the resulting change in logit difference.

## Preliminary Results

The following results were obtained on the current clean–corrupted prompt pair. They are preliminary and have not yet been validated across a broader prompt set.

### Individual Latent Intervention

| Rank | SAE Latent | Logit-Difference Recovery |
| ---: | ---------: | ------------------------: |
|    1 |      13484 |                    42.09% |
|    2 |      15707 |                     5.97% |
|    3 |      16308 |                     3.81% |
|    4 |       8876 |                     1.51% |
|    5 |      12354 |                     0.97% |

Latent `13484` produced the largest measured recovery among the individually tested features.

### Combined Latent Intervention

| Features Combined | Recovery |
| ----------------: | -------: |
|             Top 1 |   42.09% |
|             Top 2 |   46.41% |
|             Top 3 |   47.93% |
|             Top 4 |   48.91% |
|             Top 5 |   49.71% |

The top five features jointly recovered 49.71% of the clean–corrupted logit-difference gap in this experiment. This suggests that the selected features account for part of the measured behavioral difference, but do not fully recover it.

### Latent Ablation

| SAE Latent | Change in Logit Difference |
| ---------: | -------------------------: |
|      13484 |                     2.0430 |
|      15707 |                     0.2462 |
|      16308 |                    -0.1100 |
|       8876 |                    -0.0382 |
|      12354 |                     0.0233 |

Latent `13484` produced the largest measured change in logit difference during ablation. The other tested features had substantially smaller effects.

These measurements provide preliminary evidence that the tested features differ in their behavioral effects. They do not establish that the five features constitute a complete causal circuit or that the findings generalize to other prompts.

## Repository Structure

```text
jumprelu-sae-faithfulness-project/
├── main.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── patching.py
│   └── prompts.py
└── README.md
```

## Setup

Python 3.10 or a compatible version supported by the installed dependencies is recommended.

Clone the repository:

```bash
git clone https://github.com/osyeboah/jumprelu-sae-faithfulness-project.git
cd jumprelu-sae-faithfulness-project
```

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running the Experiments

Run the experiment entry point from the repository root:

```powershell
python main.py
```

The script loads GPT-2 Small and the configured SAE, runs the latent intervention experiments, ranks candidate features, evaluates combinations of top-ranked features, and reports ablation measurements.

The first run may require downloading model and SAE weights.

## Limitations and Next Steps

The current findings are based on a single clean–corrupted prompt pair. Further work is needed to evaluate their robustness.

Planned next steps include:

* Validate the candidate features across multiple equivalent prompts.
* Compare results against suitable baseline interventions.
* Investigate the sensitivity of the results to intervention design.
* Examine whether the identified features generalize beyond the current examples.
* Improve the experimental analysis and document reproducible evaluation procedures.

## Research Focus

This project is an ongoing investigation into SAE feature faithfulness, causal interventions, and feature-level explanations of transformer behavior. Its conclusions will be updated as further validation experiments are completed.
