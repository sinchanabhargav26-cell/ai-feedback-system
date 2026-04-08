"""
test.py
=======
Test the Customer Support RL Environment with random and oracle agents.

What this script does:
  1. Runs N episodes with a RANDOM agent (baseline — picks actions by chance)
  2. Optionally runs N episodes with an ORACLE agent (always correct)
  3. Prints per-episode state / action / reward clearly
  4. Prints a summary with accuracy, avg reward, and action breakdown

KEY FIXES & IMPROVEMENTS over original:
  - action_space.sample() result is cast to int before passing to step()
    to avoid the numpy int64 assertion bug in the shim.
  - Summary table now shows satisfaction score average too.
  - --oracle flag runs oracle comparison automatically.
  - Output is colour-coded with emoji for quick readability.
  - Handles KeyboardInterrupt gracefully.

Run:
    python test.py
    python test.py --episodes 50
    python test.py --episodes 30 --oracle --quiet
"""

import sys
import os
import argparse
import numpy as np
from collections import defaultdict
from typing import List

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.support_env import CustomerSupportEnv
from env.task import ACTION_LABELS
from env.grader import Grader


# ─────────────────────────────────────────────────────────────────────────────
# Random Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_random_agent(n_episodes: int = 20, verbose: bool = True) -> List[float]:
    """
    Run a purely random policy for n_episodes.

    Args:
        n_episodes: Number of episodes to run
        verbose   : If True, print one line per episode

    Returns:
        List of rewards, one per episode
    """
    # render_mode=None → silent; we handle all output ourselves
    env = CustomerSupportEnv(render_mode=None)

    # ── Accumulators ─────────────────────────────────────────────────────
    rewards:          List[float] = []
    correct_flags:    List[int]   = []
    satisfaction_sum: float       = 0.0
    action_counts               = defaultdict(int)
    action_correct              = defaultdict(int)

    print("\n" + "═" * 70)
    print("   RANDOM AGENT TEST")
    print("═" * 70)
    if verbose:
        print(f"  {'Ep':>4}  {'Intent':>10}  {'Sentiment':>9}  {'Urgency':>6}  "
              f"{'Chosen Action':<24}  {'Correct':>7}  {'Reward':>7}")
        print("  " + "─" * 66)

    for ep in range(1, n_episodes + 1):
        # ── Reset: sample a new query ─────────────────────────────────────
        obs, info = env.reset()

        # ── Random action: cast to int to avoid numpy int64 issues ────────
        action = int(env.action_space.sample())

        # ── Step ──────────────────────────────────────────────────────────
        _, reward, _, _, step_info = env.step(action)

        # ── Metrics ───────────────────────────────────────────────────────
        is_correct = step_info.get("is_correct", False)
        rewards.append(float(reward))
        correct_flags.append(1 if is_correct else 0)
        satisfaction_sum += float(step_info.get("satisfaction_score", 0.0))
        action_counts[action] += 1
        if is_correct:
            action_correct[action] += 1

        # ── Per-episode row ────────────────────────────────────────────────
        if verbose:
            icon = "✅" if is_correct else "❌"
            chosen_short = ACTION_LABELS[action][:22]
            print(f"  {ep:>4}  {info['intent']:>10}  {info['sentiment']:>9}  "
                  f"{info['urgency']:>6}  {chosen_short:<24}  "
                  f"{icon}  {reward:>+6.1f}")

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────
    n      = len(rewards)
    acc    = sum(correct_flags) / n * 100
    avg_r  = float(np.mean(rewards))
    tot_r  = float(np.sum(rewards))
    avg_s  = satisfaction_sum / n

    print("\n" + "═" * 70)
    print("  RANDOM AGENT SUMMARY")
    print("═" * 70)
    print(f"  Episodes          : {n}")
    print(f"  Correct           : {sum(correct_flags)} / {n}  ({acc:.1f}% accuracy)")
    print(f"  Avg Reward        : {avg_r:+.3f}")
    print(f"  Total Reward      : {tot_r:+.1f}")
    print(f"  Avg Satisfaction  : {avg_s:.3f} / 1.00")
    print()
    print(f"  {'Action':<38}  {'Used':>6}  {'Correct':>8}  {'Accuracy':>9}")
    print("  " + "─" * 65)
    for a in range(4):
        cnt  = action_counts[a]
        if cnt == 0:
            continue
        corr = action_correct[a]
        a_acc = corr / cnt * 100
        bar = "█" * int(a_acc / 10)
        print(f"  [{a}] {ACTION_LABELS[a]:<34}  {cnt:>6}  {corr:>8}  "
              f"{a_acc:>8.1f}%  {bar}")
    print("═" * 70)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Oracle Agent (always correct)
# ─────────────────────────────────────────────────────────────────────────────

def run_oracle_agent(n_episodes: int = 20) -> List[float]:
    """
    Run a perfect oracle agent that always selects the ground-truth action.

    Useful for:
      - Verifying the reward logic gives positive rewards for correct actions
      - Computing the theoretical maximum reward as a benchmark

    Returns:
        List of rewards (should all be positive)
    """
    env = CustomerSupportEnv(render_mode=None)
    rewards: List[float] = []

    print("\n" + "═" * 70)
    print("   ORACLE AGENT TEST  (always correct action)")
    print("═" * 70)
    print(f"  {'Ep':>4}  {'Correct Action':<36}  {'Intent':>10}  {'Reward':>7}")
    print("  " + "─" * 62)

    for ep in range(1, n_episodes + 1):
        obs, info   = env.reset()
        correct_act = int(info["correct_action"])     # always pick correct

        _, reward, _, _, step_info = env.step(correct_act)
        rewards.append(float(reward))

        print(f"  {ep:>4}  [{correct_act}] {ACTION_LABELS[correct_act]:<33}  "
              f"{info['intent']:>10}  {reward:>+6.1f}")

    env.close()

    avg_r = float(np.mean(rewards))
    tot_r = float(np.sum(rewards))
    print(f"\n  Oracle Avg Reward : {avg_r:+.3f}")
    print(f"  Oracle Total      : {tot_r:+.1f}")
    print("═" * 70)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Trained Agent Test
# ─────────────────────────────────────────────────────────────────────────────

def run_trained_agent(n_episodes: int = 20, qtable_path: str = "qtable.json") -> List[float]:
    """
    Run the trained Q-Learning agent (loaded from qtable.json) in greedy mode.

    Args:
        n_episodes  : Number of evaluation episodes
        qtable_path : Path to saved Q-table JSON

    Returns:
        List of rewards, or empty list if no Q-table found.
    """
    full_path = os.path.join(_PROJECT_ROOT, qtable_path)
    if not os.path.exists(full_path):
        print(f"\n  [Trained Agent] No Q-table found at '{full_path}'.")
        print("  Run: python train.py  to train the agent first.")
        return []

    from train import QLearningAgent
    agent = QLearningAgent(n_actions=4, epsilon_start=0.0)
    agent.load(full_path)
    agent.epsilon = 0.0   # greedy — no exploration

    env = CustomerSupportEnv(render_mode=None)
    rewards:       List[float] = []
    correct_flags: List[int]   = []

    print("\n" + "═" * 70)
    print(f"   TRAINED AGENT TEST  (greedy, Q-table from '{qtable_path}')")
    print("═" * 70)
    print(f"  {'Ep':>4}  {'Intent':>10}  {'Chosen Action':<24}  "
          f"{'Correct':>7}  {'Reward':>7}")
    print("  " + "─" * 58)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        action    = int(agent.select_action(obs))

        _, reward, _, _, step_info = env.step(action)
        is_correct = step_info.get("is_correct", False)
        rewards.append(float(reward))
        correct_flags.append(1 if is_correct else 0)

        icon = "✅" if is_correct else "❌"
        print(f"  {ep:>4}  {info['intent']:>10}  "
              f"{ACTION_LABELS[action][:22]:<24}  {icon}  {reward:>+6.1f}")

    env.close()

    acc   = sum(correct_flags) / len(correct_flags) * 100
    avg_r = float(np.mean(rewards))
    print(f"\n  Trained Agent Accuracy : {acc:.1f}%")
    print(f"  Trained Agent Avg Rwd  : {avg_r:+.3f}")
    print("═" * 70)

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Test CustomerSupportEnv with random / oracle / trained agents"
    )
    p.add_argument("--episodes", type=int,  default=20,
                   help="Number of episodes per agent (default: 20)")
    p.add_argument("--oracle",   action="store_true",
                   help="Also run the oracle (perfect) agent")
    p.add_argument("--trained",  action="store_true",
                   help="Also run the trained Q-Learning agent (needs qtable.json)")
    p.add_argument("--quiet",    action="store_true",
                   help="Suppress per-episode rows; show summary only")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        random_rewards = run_random_agent(
            n_episodes=args.episodes,
            verbose=not args.quiet,
        )

        if args.oracle:
            oracle_rewards = run_oracle_agent(n_episodes=args.episodes)
            print(f"\n  Comparison → Random avg: {np.mean(random_rewards):+.3f}  "
                  f"vs  Oracle avg: {np.mean(oracle_rewards):+.3f}")

        if args.trained:
            trained_rewards = run_trained_agent(n_episodes=args.episodes)
            if trained_rewards:
                print(f"\n  Comparison → Random avg: {np.mean(random_rewards):+.3f}  "
                      f"vs  Trained avg: {np.mean(trained_rewards):+.3f}")

    except KeyboardInterrupt:
        print("\n\n  [test.py] Interrupted by user.")
        sys.exit(0)
