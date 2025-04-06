# AFA-Benchmark

Methods to compare:

Generative greedy:

1. Chao Ma, et al. Eddi: Efficient dynamic discovery of high-value information with partial VAE. ICML 2019.
2. Wenbo Gong, et al. Icebreaker: Element-wise efficient information acquisition with a bayesian deep latent gaussian model. NeurIPS 2019.

Discriminative greedy:

3. Ian Connick Covert, et al. Learning to maximize mutual information for dynamic feature selection. ICML 2023.
4. Soham Gadgil, et al. Estimating conditional mutual information for dynamic feature selection. ICML 2024.

RL:

5. Hajin Shim, et al. Joint active feature acquisition and classification with variable-size set
encoding. NeurIPS 2018.
6. Jaromír Janisch, et al. Classification with costly features using deep reinforcement learning. AAAI 2019.
7. Sara Zannone, et al. Odin: Optimal discovery of high-value information using model-based deep reinforcement learning. ICML workshop, 2019.
8. Yang Li and Junier Oliva. Active feature acquisition with generative surrogate models. ICML 2021.


Maybe:

9. Mohammad Kachuee, et al. Opportunistic learning: Budgeted cost-sensitive learning from data streams. ICLR, 2019.)
10. Gabriel Dulac-Arnold, et al. Datum-wise classification: a sequential approach to sparsity. ECML PKDD 2011.
11. Thomas Rückstieß, et al. Sequential feature selection for classification. AI 2011.
12. Yang Li and Junier Oliva. Distribution guided active feature acquisition, arxiv 2024. **(Journal extension of Li and Oliva 2021).**
13. He He, et al. Imitation learning by coaching. NeurIPS 2012
14. Jaromír Janisch, et al. Classification with costly features as a sequential decision-making problem. MLJ 2020. **(Journal extension of Janisch et al 2019).**
15. Aditya Chattopadhyay, et al. Variational information pursuit for interpretable predictions. ICLR, 2023.
16. Samrudhdhi B Rangrej et al. A probabilistic hard attention model for sequentially observed scenes. BMVC 2021. **(Extends EDDI to image data).**
17. Ghosh et al. DiFA: Differentiable Feature Acquisition. AAAI 2023.

## State requirements per method at test time

All of the four RL methods above need two things at test time:
- Currently selected features as a vector, with $0$ for non-acquired features.
- Boolean feature mask. $1$ if feature is acquired, $0$ if not.

## Synthetic dataset generation

Things to consider for synthetic data generation in the context of AFA (extension of CUBE):

- Number of features.
- Number of classes.
- Number of informative/redundant features for each class.
- Degree of overlap in which features are relevant for each class.
- Cost profiles (uniform vs. highly skewed).
- Label noise
- Class balance
- Synthetic data where non-greedy selection is better than greedy

