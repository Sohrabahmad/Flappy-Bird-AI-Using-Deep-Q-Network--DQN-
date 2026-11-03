# 🐦 Flappy Bird AI Using Deep Q-Network (DQN)

## 📌 Project Overview

This project implements an AI-powered Flappy Bird agent using **Deep Q-Network (DQN)**, a value-based Reinforcement Learning algorithm.

The agent learns to play Flappy Bird by interacting with the game environment, collecting rewards, and improving its decisions through a neural network. It uses Experience Replay and a Target Network to make the learning process more stable.

The project also includes a manual gameplay script that allows users to control the bird using the keyboard.

## 🎯 Project Objectives

- Understand the fundamentals of Reinforcement Learning.
- Implement Deep Q-Learning using PyTorch.
- Train an AI agent to play Flappy Bird.
- Use Experience Replay to improve training efficiency.
- Implement a Policy Network and Target Network.
- Apply epsilon-greedy action selection for exploration and exploitation.
- Save and load trained model weights.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning and neural networks |
| Gymnasium | Reinforcement Learning environment |
| Flappy Bird Gymnasium | Flappy Bird game environment |
| NumPy | Numerical operations |
| Pygame | Keyboard input and game interaction |
| PyYAML | Hyperparameter configuration |
| Random | Exploration and experience sampling |

## 📂 Project Structure

```text
Flappy-Bird-DQN/
│
├── agent.py                # DQN agent and training logic
├── dqn.py                  # Deep Q-Network architecture
├── experience_replay.py    # Experience Replay memory
├── game_flappy_bird.py     # Manual Flappy Bird gameplay
├── parameters.yaml         # Hyperparameter configuration (required)
│
├── runs/                   # Generated training outputs
│   ├── *.log               # Best reward logs
│   └── *.pt                # Saved model weights
│
└── README.md
```

*Note: `parameters.yaml` is required by `agent.py`, but it was not included in the uploaded files. The `runs/` directory and its contents are generated during training.*

## 🧠 Deep Q-Network Architecture

The project uses a fully connected neural network implemented with PyTorch.

**Network architecture:**

```text
       Input Layer
       State Features
             |
             v
       Hidden Layer
       256 Neurons
             |
           ReLU
             |
             v
        Output Layer
        2 Q-Values
             |
             v
       Action Selection
```

The network receives the environment's state as input and predicts Q-values for the available actions.

The action with the highest predicted Q-value is selected during exploitation.

### Network Configuration

- Input dimension: Determined by the environment observation space.
- Hidden layer: 256 neurons.
- Activation function: ReLU.
- Output dimension: Determined by the environment action space (2 actions in the game).

## 🔄 Reinforcement Learning Workflow

The agent follows these steps:

1. Initialize the Flappy Bird environment.
2. Observe the current game state.
3. Select an action using epsilon-greedy policy.
4. Perform the selected action.
5. Receive the next state, reward, and termination status.
6. Store the experience in Replay Memory.
7. Sample a mini-batch of previous experiences.
8. Calculate target Q-values using the Target Network.
9. Update the Policy Network using backpropagation.
10. Synchronize the Target Network periodically.
11. Save the best-performing model.

## 📚 Experience Replay

Experience Replay stores previous interactions in a FIFO queue using Python's `deque`.

Each experience contains:

```python
(state, action, next_state, reward, terminated)
```

The agent randomly samples a mini-batch from the memory buffer to train the neural network.

**Benefits:**
- Reduces correlation between consecutive experiences.
- Improves learning stability.
- Allows previously collected experiences to be reused.
- Makes training more efficient.

## ⚙️ Deep Q-Learning Update

The project calculates target Q-values using the Bellman equation:

\[
Q_{\text{target}} =
r + (1-d)\gamma \max_{a'}Q_{\text{target}}(s',a')
\]

Where:

- \(r\) = Reward received.
- \(d\) = Termination indicator.
- \(\gamma\) = Discount factor.
- \(s'\) = Next state.
- \(a'\) = Possible next action.

The model uses Mean Squared Error (MSE) loss to compare predicted Q-values with target Q-values.

## 🎮 Action Selection

The project uses an epsilon-greedy strategy.

- **Exploration:** Select a random action with probability epsilon.
- **Exploitation:** Select the action with the highest predicted Q-value.

The epsilon value gradually decreases during training, encouraging the agent to move from exploration toward exploitation.

## 🕹️ Manual Gameplay

The `game_flappy_bird.py` file allows users to play Flappy Bird manually.

**Controls:**

| Key | Action |
|---|---|
| SPACE | Flap / Jump |
| Window Close | Exit the game |

The default action is `0` (no flap), while action `1` makes the bird flap.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Flappy-Bird-DQN.git
cd Flappy-Bird-DQN
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml numpy
```

## 🚀 How to Run

### Run Manual Gameplay

```bash
python game_flappy_bird.py
```

Press SPACE to make the bird flap.

### Train the DQN Agent

```bash
python agent.py <parameter_set> --train
```

Replace `<parameter_set>` with a configuration key defined in `parameters.yaml`.

### Test the Trained Agent

```bash
python agent.py <parameter_set>
```

The testing mode loads the saved model weights and renders the game.

## ⚙️ Hyperparameter Configuration

The agent reads its hyperparameters from `parameters.yaml`.

The following parameters are required:

| Parameter | Description |
|---|---|
| `alpha` | Learning rate |
| `gamma` | Discount factor |
| `epsilon_init` | Initial exploration probability |
| `epsilon_min` | Minimum exploration probability |
| `epsilon_decay` | Epsilon decay rate |
| `replay_memory_size` | Maximum replay buffer capacity |
| `mini_batch_size` | Number of experiences sampled per update |
| `reward_threshold` | Episode reward stopping threshold |
| `network_sync_rate` | Target network synchronization interval |

The actual parameter values depend on the configuration selected when running the agent.

## 💾 Model Saving and Logging

During training, the project creates a `runs/` directory.

It stores:

- **Model files (`.pt`):** Saved Policy Network weights.
- **Log files (`.log`):** Best reward and corresponding episode information.

The best-performing model is saved whenever a new best episode reward is achieved.

## 🔍 Key Concepts Learned

- Reinforcement Learning
- Deep Q-Network (DQN)
- Q-Learning and Bellman Equation
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Neural Network Optimization
- PyTorch Model Training
- Gymnasium Environment Interaction

## 🚧 Future Improvements

- Add reward and episode performance visualization.
- Implement Double DQN to reduce Q-value overestimation.
- Add Dueling DQN architecture.
- Improve reward shaping for faster learning.
- Add training checkpoints and resume functionality.
- Evaluate performance over multiple episodes.
- Fix and verify periodic Target Network synchronization.
- Handle both `terminated` and `truncated` environment outcomes.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📜 License

This project is intended for educational and learning purposes. You can modify and extend it for your own experiments.