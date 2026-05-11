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



### Requirements

There is no formal installation script. To run the code, ensure you have the following in your environment:

* **PyTorch** (with CUDA for GPU acceleration)
* **NumPy**
* **TensorboardX** (for logging)
* **`utils.py`**: You must have a local `utils.py` file containing the `TargetNet` class (used for soft-updating target networks).

### 2. File Structure

Place the provided script in your project directory. Your folder should look like this:

```text
project/
├── utils.py          # Contains TargetNet
├── sac_regulator.py  # The provided SAC + SACREG code
└── main.py           # Your training loop

```

### A. Initializing the Agent

In your main script, initialize the `SAC` class by providing your environment's dimensions and a replay buffer.

```python
from sac_regulator import SAC

# Initialize agent
agent = SAC(
    seed=0,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    replay_buffer=my_buffer,
    batch_size=256,
    name="Humanoid_Exp",
    env_id="EnvV1",
    date="2026-05-11"
)

```

### B. Inference (In the Loop)

The `select_action` method handles the regulation automatically. It samples an action from the actor and scales it via the regulator network (`SACREG`) before returning it.

```python
# During interaction:
action, predicted_cost = agent.select_action(state, eval=False)
next_state, reward, done, info = env.step(action)

```

### C. Training

The `train()` function performs a single update step for all networks (Actor, Twin Q-Nets, Twin C-Nets, and the Regulator).



### Hyperparameter Tuning

To get the code working for your specific environment, tune these two parameters at the top of the script:

| Parameter | Default | Description |
| --- | --- | --- |
| `COSTS_PEN` | `20` | **Safety Weight.** Higher values lead to more aggressive scaling (smaller actions) to avoid costs. |
| `MATCH_PEN` | `0.01` | **Action Matching.** Encourages the regulator to keep scaling factors close to 1.0 (prevents action collapse). |
| `COST_THRESHOLD` | `10` | The maximum cost value the `TwinCNets` are trained to predict. |

### Example Training Loop

```python
for step in range(TOTAL_STEPS):
    # 1. Get regulated action
    action, _ = agent.select_action(state)
    
    # 2. Step environment
    next_state, reward, done, info = env.step(action)
    cost = info.get('cost', 0.0) # Ensure your env provides this!
    
    # 3. Store in buffer
    replay_buffer.add(state, action, next_state, reward, float(not done), cost)
    
    # 4. Update networks
    if step > START_STEPS:
        agent.train()

```



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
