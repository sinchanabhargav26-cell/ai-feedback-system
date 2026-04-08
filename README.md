# 🤖 Customer Support Decision Agent with Escalation Intelligence

> A production-quality Reinforcement Learning environment where an agent learns
> to optimally triage customer support queries — built with Gymnasium and Q-Learning.

---

## 📌 Project Overview

Customer support teams handle hundreds of queries daily. Every query needs a
decision: can it be auto-answered? Does it need clarification? Must it be
escalated to a human? Making the wrong call costs money (unnecessary escalation),
satisfaction (unresolved critical issues), or both.

This project frames that decision as a **Reinforcement Learning problem**:

- The **environment** presents a customer query as a structured state
- The **agent** picks one of four discrete actions
- A **grader** evaluates the decision and assigns a shaped reward
- The agent learns over thousands of episodes to maximise total reward

---

## 🧠 How Reinforcement Learning is Used

```
         ┌─────────────────────────────────────────────┐
         │           CustomerSupportEnv                 │
         │                                              │
  reset()│  state = [intent, sentiment, urgency]        │
    ──►  │                                              │
         │  obs = encode(state)  →  float32[3]          │
         └──────────────────────┬──────────────────────┘
                                │ observation
                                ▼
                    ┌───────────────────────┐
                    │    Q-Learning Agent   │
                    │                       │
                    │  ε-greedy policy:     │
                    │  explore OR exploit   │
                    │  Q-table[(s)] → a     │
                    └───────────┬───────────┘
                                │ action ∈ {0,1,2,3}
                                ▼
         ┌─────────────────────────────────────────────┐
         │           Grader Module                      │
         │                                              │
  step() │  grade(task, action) → GradeResult           │
    ◄──  │  reward = base_reward + cost_penalty         │
         │  satisfaction_score ∈ [0, 1]                 │
         └─────────────────────────────────────────────┘
```

**Algorithm: Tabular Q-Learning**

The state space is small: 4 × 3 × 3 = **36 possible states**.
This makes tabular Q-learning ideal — no neural network needed.

Update rule (Bellman equation):
```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') − Q(s, a)]
```

Since episodes are single-step, `terminated=True` → future term = 0 → simplifies to:
```
Q(s, a) ← Q(s, a) + α × [r − Q(s, a)]
```

**Exploration: ε-greedy decay**
```
ε starts at 1.0 (fully random) → decays × 0.995/episode → floors at 0.05
```

---

## 🎮 Environment Design

### Observation Space
`Box(3,)` — three categorical features encoded as integers:

| Index | Feature   | Values                                      |
|-------|-----------|---------------------------------------------|
| 0     | intent    | billing=0, technical=1, general=2, complaint=3 |
| 1     | sentiment | positive=0, neutral=1, angry=2              |
| 2     | urgency   | low=0, medium=1, high=2                     |

### Action Space
`Discrete(4)` — four possible triage decisions:

| Action | Label | When to use |
|--------|-------|-------------|
| **0** | Auto-respond | Query has a known, documented solution |
| **1** | Ask for clarification | Query is ambiguous or incomplete |
| **2** | Escalate to human | Critical, complex, or angry customer |
| **3** | Fallback response | No direct solution; acknowledge and redirect |

### Episode Structure
Each episode is **single-step**:
1. `reset()` → samples a random query from the dataset → returns observation
2. Agent chooses action
3. `step(action)` → grades action → returns `(obs, reward, terminated=True, truncated=False, info)`

---

## 💰 Reward Function

All reward constants live in `env/grader.py` — change them in one place.

| Scenario | Reward |
|:---------|-------:|
| ✅ Correct escalation (high-urgency query routed to human) | **+3.0** |
| ✅ Correct auto-respond or correct fallback | **+2.0** |
| ✅ Correct clarification on ambiguous query | **+1.0** |
| ❌ Wrong action (not the most dangerous case) | **−2.0** |
| ⚠️ Unnecessary escalation (simple query over-escalated) | **−1.0** |
| ⚠️ Fallback overused (better action was available) | **−1.0** |
| 🚨 **Critical issue NOT escalated** (angry + high urgency) | **−5.0** |
| 💸 Operational cost of escalation | **−0.5** |

**Simulated satisfaction score** (0.0–1.0) is returned alongside reward:
- Correct escalation → 0.85 | Correct action → 0.90 | Clarification → 0.70
- Failed escalation → 0.10 | Incorrect action → 0.25

---

## 🏗️ Project Structure

```
project/
├── env/
│   ├── __init__.py        # Clean package exports (all symbols)
│   ├── support_env.py     # Gymnasium Env — reset(), step(), render()
│   ├── task.py            # Task dataclass, encoding maps, TaskLoader
│   └── grader.py          # GradeResult, Grader, reward constants
├── data/
│   └── sample_queries.json  # 20 labelled queries (all 4 intents covered)
├── gym_shim.py            # Fallback when gymnasium is not installed
├── train.py               # QLearningAgent + training loop + evaluate()
├── test.py                # Random / oracle / trained agent testing
├── app.py                 # Gradio demo UI for Hugging Face Spaces
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Offline / no-internet?** The project includes `gym_shim.py` which
> provides a built-in Gymnasium-compatible fallback. Everything except
> `app.py` works without installing `gymnasium` or `gradio`.

### 2. Test the environment (random agent)

```bash
python test.py
```

Expected output: episode table with intent / sentiment / chosen action / reward,
followed by an accuracy and reward summary.

```bash
# More episodes, oracle comparison, trained-agent comparison
python test.py --episodes 50 --oracle --trained
```

### 3. Train the Q-Learning agent

```bash
python train.py
```

Trains for 1,000 episodes and saves `qtable.json`. Typical results:
- **Random baseline**: ~30% accuracy, avg reward ~−1.7
- **After 1,000 episodes**: ~85–93% accuracy, avg reward ~+1.8

```bash
# Custom hyperparameters
python train.py --episodes 2000 --lr 0.1 --eps-decay 0.998 --eval 300
```

### 4. Launch the Gradio demo

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

---

## 🌐 Deploy to Hugging Face Spaces

1. Create a free account at https://huggingface.co
2. Go to **Spaces → Create new Space → Gradio SDK**
3. Upload all project files (or push via Git):

```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Initial commit"
git push
```

4. Make sure `app.py` is at the root level (Gradio's default entry point)
5. `requirements.txt` is installed automatically by Spaces
6. Upload `qtable.json` (trained Q-table) for best demo results

---

## 🧪 Dataset

`data/sample_queries.json` contains **20 labelled queries** covering:

| Intent    | Count | Examples |
|-----------|-------|---------|
| billing   | 7     | double-charge, invoice, refund, fraud |
| technical | 6     | password reset, WiFi, app crash, data loss |
| general   | 4     | business hours, vague queries, feedback |
| complaint | 3     | cancellation, poor service, escalation requests |

Each query has:
- `intent`, `sentiment`, `urgency` — state features
- `correct_action` — ground-truth label (0–3)
- `expected_outcome` — what a human agent would do
- `notes` — explains why that action is correct

---

## 🐛 Bugs Fixed (vs original codebase)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `gym_shim.py` | `_Env.reset()` raised `NotImplementedError` on `super().reset()` | Base `reset()` now sets seed and returns cleanly |
| 2 | `train.py` | `obs_to_state()` stored `numpy.int64` keys — subtle hash mismatch | Explicit `int()` cast on all state key components |
| 3 | `app.py` | `predict()` had no `try/except` — any internal error showed as "Error" | Full try/except; always returns 7-string tuple |
| 4 | `app.py` | Gradio `outputs` list order didn't match `predict()` return order | Outputs list reordered to match exactly |
| 5 | `env/grader.py` | `GradeResult.to_dict()` leaked `numpy.bool_` / `numpy.float64` types | All values cast to Python native types |
| 6 | `env/task.py` | No validation — bad intent/sentiment caused silent `KeyError` | `__post_init__` validates all fields with clear error messages |
| 7 | `env/task.py` | `OBS_LOW/OBS_HIGH` not exported — `support_env.py` had hardcoded values | Derived from maps; exported; imported in `support_env.py` |
| 8 | `env/support_env.py` | `decode_observation()` rebuilt inverse maps on every call | Uses `INTENT_INV`/`SENTIMENT_INV`/`URGENCY_INV` from `task.py` |
| 9 | `app.py` | Heuristic policy only covered 18/36 states | Covers all 36 states via systematic rule generation |
| 10 | `env/__init__.py` | Did not export reward constants | All constants now exported |

---

## 📊 Performance Benchmarks

| Agent | Episodes | Accuracy | Avg Reward |
|-------|----------|----------|------------|
| Random (baseline) | — | ~30% | ~−1.7 |
| Heuristic (expert rules) | — | ~75% | ~+1.2 |
| Q-Learning (trained) | 1,000 | ~85% | ~+1.8 |
| Q-Learning (trained) | 2,000 | ~90% | ~+1.9 |
| Oracle (perfect) | — | 100% | ~+2.0 |

---

## 📄 License

MIT — free to use, modify, and distribute for any purpose.
