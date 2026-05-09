# Constraint-Aware-Adaptive-Action-Scaling
Safe Reinforcement Learning (RL) often struggles to balance task performance with safety constraints, frequently leading to instability or overly conservative behavior. This repository implements a modular, cost-aware regulator that decouples reward maximization from safety enforcement.  Instead of overriding the RL agent or using complex dual updates, our approach uses Adaptive Action Scaling. The regulator learns to smoothly attenuate potentially unsafe actions by element-wise scaling based on predicted future constraint violations. 

## Implementation Details

**Decoupled Objectives**: Reward maximization (`SACActor`) is separated from safety enforcement (`SACREG`) to prevent conflicting gradients.
**Twin Critics**: The framework utilizes `TwinQNets` for task rewards and `TwinCNets` for conservative, robust cost estimation.


### Action Modulation

**Adaptive Scaling**: Raw actions are modulated via element-wise multiplication: $\tilde{a}_{t}=\rho_{t}\odot a_{t}$.
**Scaling Vector**: The regulator outputs a vector $\rho \in (0, 1]^d$ using sigmoid activation, allowing fine-grained control over specific risky joints.


### Optimization Strategy

**Regulator Loss**: The training objective balances predicted cost minimization ($Q_c$) with a logarithmic barrier penalty ($-\log \rho$) to prevent action collapse.
**Stable Training**: Updates for both reward and cost critics are based on the actual executed (regulated) actions to ensure accurate off-policy learning.
**Gradient Isolation**: Scaling weights are detached during critic updates to maintain clean modularity between task and safety modules.


This repository contains the code for the regulator presented in the paper **"Constraint-Aware Reinforcement Learning via Adaptive Action Scaling"**, accepted at the **8th Annual Learning for Dynamics & Control Conference (L4DC 2026)**.

**Paper:** [https://arxiv.org/abs/2510.11491](https://arxiv.org/abs/2510.11491)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dawood2026L4DC,
  title={Constraint-Aware Reinforcement Learning via Adaptive Action Scaling},
  author={Dawood, Murad and Siddiquie, Usama Ahmed and Khorshidi, Shahram and Bennewitz, Maren},
  booktitle={Proc. of the Learning for Dynamics \& Control Conference (L4DC)},
  year={2026}
}
```
