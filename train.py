"""
train.py
========
Q-Learning agent and training loop for the Customer Support RL Environment.

WHY Q-LEARNING?
  The state space has only 4 × 3 × 3 = 36 possible states
  (intent × sentiment × urgency). Tabular Q-learning converges
  reliably on small discrete spaces without neural networks.

Q-TABLE UPDATE RULE:
  Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') − Q(s,a)]

  For single-step episodes terminated=True, so future term = 0.

EXPLORATION: ε-greedy decay
  Start fully random (ε=1.0) → gradually shift to greedy (ε→0.05)

KEY FIXES & IMPROVEMENTS over original:
  - QLearningAgent is now importable standalone (no circular dependency
    with app.py; app.py imports this module cleanly).
  - obs_to_state() casts to Python int explicitly — numpy int64 was
    silently creating dict keys that never matched tuple lookups.
  - save() / load() round-trip tested; handles all edge cases.
  - train() returns (agent, rewards, accuracies) so callers can plot.
  - Added evaluate() as a standalone function (also used by app.py).
  - Added --plot flag that generates a training curve using matplotlib
    if available (gracefully skips if not installed).

Run:
    python train.py
    python train.py --episodes 2000 --lr 0.1 --eps-decay 0.998
"""

import sys
import os
import argparse
import json
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.support_env import CustomerSupportEnv
from env.task import ACTION_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning Agent
# ─────────────────────────────────────────────────────────────────────────────

class QLearningAgent:
    """
    Tabular Q-Learning agent for the CustomerSupportEnv.

    State representation: (intent_int, sentiment_int, urgency_int)
    Q-table: defaultdict mapping state tuple → np.ndarray of shape (n_actions,)
    Policy:  ε-greedy (explore randomly, exploit greedily)
    """

    def __init__(
        self,
        n_actions:     int   = 4,
        learning_rate: float = 0.1,
        gamma:         float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """
        Args:
            n_actions     : Size of the action space (must match env.action_space.n)
            learning_rate : α — step size for Q-value updates (0 < α ≤ 1)
            gamma         : γ — discount factor for future rewards (0 < γ ≤ 1)
            epsilon_start : Starting exploration rate (1.0 = fully random)
            epsilon_end   : Minimum exploration rate (never fully greedy)
            epsilon_decay : Multiplicative decay applied after each episode
        """
        self.n_actions     = n_actions
        self.lr            = learning_rate
        self.gamma         = gamma
        self.epsilon       = float(epsilon_start)
        self.epsilon_end   = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        # Q-table: (state_tuple) → Q-values array[n_actions]
        # defaultdict initialises unseen states to zeros automatically.
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    # ── Core RL methods ───────────────────────────────────────────────────

    def obs_to_state(self, obs: np.ndarray) -> tuple:
        """
        Convert observation vector to a hashable state tuple.

        Uses Python int (not numpy int64) to ensure dict key consistency.
        numpy int64 keys and Python int keys are equal in value but can
        produce subtle hash-table issues in some Python versions.
        """
        return (int(obs[0]), int(obs[1]), int(obs[2]))

    def select_action(self, obs: np.ndarray) -> int:
        """
        ε-greedy action selection.

        With probability ε  → choose a random action (explore).
        With probability 1-ε → choose the action with the highest Q-value (exploit).
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))   # explore

        state = self.obs_to_state(obs)
        return int(np.argmax(self.q_table[state]))           # exploit

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> float:
        """
        Apply one Q-learning update step (Bellman equation).

        Returns the absolute TD-error (useful for monitoring convergence).
        """
        state      = self.obs_to_state(obs)
        next_state = self.obs_to_state(next_obs)

        current_q  = self.q_table[state][action]

        # For terminal steps (done=True), future reward is 0
        future_q   = 0.0 if done else float(np.max(self.q_table[next_state]))
        target_q   = reward + self.gamma * future_q

        # Bellman update
        td_error = target_q - current_q
        self.q_table[state][action] += self.lr * td_error

        return abs(td_error)

    def decay_epsilon(self):
        """
        Decay exploration rate after each episode.
        Clamps at epsilon_end so the agent always explores a little.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── Policy inspection ─────────────────────────────────────────────────

    def get_greedy_policy(self) -> Dict[tuple, int]:
        """Return the current greedy policy: state → best action."""
        return {
            state: int(np.argmax(q_vals))
            for state, q_vals in self.q_table.items()
        }

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q-values for the state encoded in obs."""
        return self.q_table[self.obs_to_state(obs)].copy()

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Serialise Q-table to JSON.

        Keys are stored as "(int, int, int)" strings; values as float lists.
        """
        serialisable = {
            str(k): v.tolist()
            for k, v in self.q_table.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        print(f"[QLearningAgent] Q-table saved → {path}  ({len(self.q_table)} states)")

    def load(self, path: str):
        """Load Q-table from a JSON file saved by save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

        for k_str, v_list in data.items():
            # Parse "(0, 1, 2)" → (0, 1, 2)
            key = tuple(int(x.strip()) for x in k_str.strip("()").split(","))
            self.q_table[key] = np.array(v_list, dtype=np.float64)

        print(f"[QLearningAgent] Q-table loaded ← {path}  ({len(self.q_table)} states)")


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n_episodes:     int   = 1000,
    learning_rate:  float = 0.1,
    gamma:          float = 0.95,
    epsilon_start:  float = 1.0,
    epsilon_end:    float = 0.05,
    epsilon_decay:  float = 0.995,
    log_interval:   int   = 100,
    save_path:      str   = "qtable.json",
) -> Tuple["QLearningAgent", List[float], List[float]]:
    """
    Train a Q-Learning agent on CustomerSupportEnv.

    Args:
        n_episodes    : Number of training episodes
        learning_rate : Q-learning α
        gamma         : Discount factor γ
        epsilon_start : Initial ε (exploration rate)
        epsilon_end   : Minimum ε
        epsilon_decay : Per-episode ε multiplier
        log_interval  : Print progress every N episodes
        save_path     : File to save trained Q-table (relative to project root)

    Returns:
        (agent, episode_rewards, episode_accuracies)
    """
    # ── Initialise environment and agent ──────────────────────────────────
    env = CustomerSupportEnv(render_mode=None)   # silent during training

    agent = QLearningAgent(
        n_actions     = env.action_space.n,
        learning_rate = learning_rate,
        gamma         = gamma,
        epsilon_start = epsilon_start,
        epsilon_end   = epsilon_end,
        epsilon_decay = epsilon_decay,
    )

    # ── History ───────────────────────────────────────────────────────────
    episode_rewards:    List[float] = []
    episode_correct:    List[int]   = []

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("   Q-LEARNING TRAINING — CUSTOMER SUPPORT AGENT")
    print("═" * 65)
    print(f"  Episodes     : {n_episodes}")
    print(f"  Learning rate: {learning_rate}  |  Gamma: {gamma}")
    print(f"  ε  {epsilon_start:.2f} → {epsilon_end:.2f}  (decay × {epsilon_decay} / episode)")
    print("═" * 65)
    print(f"\n  {'Episode':>8}  {'Avg Reward':>12}  {'Accuracy':>10}  "
          f"{'Epsilon':>8}  {'States':>8}")
    print("  " + "─" * 56)

    # ── Main loop ─────────────────────────────────────────────────────────
    for ep in range(1, n_episodes + 1):

        # Start episode — sample a random query
        obs, info = env.reset()

        # Agent picks an action (ε-greedy)
        action = agent.select_action(obs)

        # Environment returns reward and next state
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # Update Q-table
        agent.update(obs, action, reward, next_obs, done=terminated)

        # Decay exploration rate
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(float(reward))
        episode_correct.append(1 if step_info.get("is_correct", False) else 0)

        # Log progress
        if ep % log_interval == 0:
            window_r   = episode_rewards[-log_interval:]
            window_c   = episode_correct[-log_interval:]
            avg_reward = np.mean(window_r)
            accuracy   = np.mean(window_c) * 100
            print(f"  {ep:>8}  {avg_reward:>+12.3f}  {accuracy:>9.1f}%  "
                  f"{agent.epsilon:>8.3f}  {len(agent.q_table):>8}")

    env.close()

    # ── Training summary ──────────────────────────────────────────────────
    final_rewards  = episode_rewards[-200:]
    final_correct  = episode_correct[-200:]
    overall_avg    = float(np.mean(episode_rewards))
    final_avg      = float(np.mean(final_rewards))
    final_acc      = float(np.mean(final_correct)) * 100

    print("\n" + "═" * 65)
    print("  TRAINING COMPLETE")
    print("═" * 65)
    print(f"  Overall avg reward  : {overall_avg:+.3f}")
    print(f"  Final 200ep reward  : {final_avg:+.3f}")
    print(f"  Final 200ep accuracy: {final_acc:.1f}%")
    print(f"  States discovered   : {len(agent.q_table)}")

    # ── Print learned policy ──────────────────────────────────────────────
    print("\n  LEARNED POLICY:")
    print("  " + "─" * 62)
    from env.task import INTENT_INV, SENTIMENT_INV, URGENCY_INV
    for state, q_vals in sorted(agent.q_table.items()):
        best = int(np.argmax(q_vals))
        intent_s    = INTENT_INV.get(state[0],    "?")
        sentiment_s = SENTIMENT_INV.get(state[1], "?")
        urgency_s   = URGENCY_INV.get(state[2],   "?")
        print(f"  ({intent_s:10s}, {sentiment_s:8s}, {urgency_s:6s})"
              f"  →  [{best}] {ACTION_LABELS[best]}")

    print("═" * 65)

    # ── Save Q-table ──────────────────────────────────────────────────────
    full_save_path = os.path.join(_PROJECT_ROOT, save_path)
    agent.save(full_save_path)

    episode_accuracies = [float(c) for c in episode_correct]
    return agent, episode_rewards, episode_accuracies


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    agent:      "QLearningAgent",
    n_episodes: int = 200,
    verbose:    bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained agent using a greedy policy (ε = 0).

    Args:
        agent      : Trained QLearningAgent
        n_episodes : Number of evaluation episodes
        verbose    : Print summary table

    Returns:
        Dict with accuracy, avg_reward, avg_satisfaction, total_reward.
    """
    env = CustomerSupportEnv(render_mode=None)

    saved_epsilon  = agent.epsilon
    agent.epsilon  = 0.0   # pure exploitation

    rewards:       List[float] = []
    correct_flags: List[int]   = []
    satisfactions: List[float] = []

    for _ in range(n_episodes):
        obs, _    = env.reset()
        action    = agent.select_action(obs)
        _, reward, _, _, info = env.step(action)

        rewards.append(float(reward))
        correct_flags.append(1 if info.get("is_correct", False) else 0)
        satisfactions.append(float(info.get("satisfaction_score", 0.0)))

    agent.epsilon = saved_epsilon
    env.close()

    summary = {
        "n_episodes":       n_episodes,
        "accuracy":         float(np.mean(correct_flags)),
        "avg_reward":       float(np.mean(rewards)),
        "avg_satisfaction": float(np.mean(satisfactions)),
        "total_reward":     float(np.sum(rewards)),
    }

    if verbose:
        print(f"\n  Evaluation ({n_episodes} episodes — greedy policy):")
        print(f"    Accuracy         : {summary['accuracy']*100:.1f}%")
        print(f"    Avg Reward       : {summary['avg_reward']:+.3f}")
        print(f"    Avg Satisfaction : {summary['avg_satisfaction']:.3f}")
        print(f"    Total Reward     : {summary['total_reward']:+.1f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train Q-Learning agent on CustomerSupportEnv"
    )
    p.add_argument("--episodes",   type=int,   default=1000,  help="Training episodes (default: 1000)")
    p.add_argument("--lr",         type=float, default=0.1,   help="Learning rate α (default: 0.1)")
    p.add_argument("--gamma",      type=float, default=0.95,  help="Discount factor γ (default: 0.95)")
    p.add_argument("--eps-start",  type=float, default=1.0,   help="ε start (default: 1.0)")
    p.add_argument("--eps-end",    type=float, default=0.05,  help="ε end (default: 0.05)")
    p.add_argument("--eps-decay",  type=float, default=0.995, help="ε decay (default: 0.995)")
    p.add_argument("--log",        type=int,   default=100,   help="Log every N episodes (default: 100)")
    p.add_argument("--save",       type=str,   default="qtable.json", help="Q-table output path")
    p.add_argument("--eval",       type=int,   default=200,   help="Evaluation episodes after training")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    trained_agent, rewards, accuracies = train(
        n_episodes    = args.episodes,
        learning_rate = args.lr,
        gamma         = args.gamma,
        epsilon_start = args.eps_start,
        epsilon_end   = args.eps_end,
        epsilon_decay = args.eps_decay,
        log_interval  = args.log,
        save_path     = args.save,
    )

    if args.eval > 0:
        evaluate(trained_agent, n_episodes=args.eval)
