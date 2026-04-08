import gradio as gr
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from support_env import SupportEnv
from grader import grade_action
env = SupportEnv()

ACTIONS = {
    0: "Auto Reply",
    1: "Ask Clarification",
    2: "Escalate",
    3: "Fallback"
}

QTABLE_FILE = "q_table.npy"
REWARD_FILE = "reward_history.npy"

# Load or initialize Q-table
if os.path.exists(QTABLE_FILE):
    q_table = np.load(QTABLE_FILE)
else:
    q_table = np.random.rand(4, 4)

# Load reward history
if os.path.exists(REWARD_FILE):
    reward_history = list(np.load(REWARD_FILE))
else:
    reward_history = []

last_state = {"value": None}
last_action = {"value": None}


# 📊 Q-values chart
def plot_q_values(q_values):
    fig, ax = plt.subplots()
    ax.bar(list(ACTIONS.values()), q_values)
    ax.set_title("Q-Values")
    return fig


# 📈 Learning graph
def plot_learning():
    fig, ax = plt.subplots()
    ax.plot(reward_history)
    ax.set_title("Learning Progress")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    return fig


# 🤖 Decision function
def get_decision(message, intent, sentiment, urgency, user_action):

    try:
        yield "🤖 Thinking...", "", "", "", "", None, None
        time.sleep(0.5)

        yield "🔍 Analyzing...", "", "", "", "", None, None
        time.sleep(0.5)

        state, _ = env.reset(intent, sentiment, urgency)
        state_idx = int(state[0])

        last_state["value"] = state_idx

        q_values = q_table[state_idx]
        ai_action = int(np.argmax(q_values))

        last_action["value"] = ai_action

        _, reward, _, _, _ = env.step(ai_action)
        grade = grade_action(ai_action, urgency)

        decision = f"🤖 {ACTIONS[ai_action]}"
        decision = f"✅ {decision}" if reward > 0 else f"❌ {decision}"

        compare = ""
        if user_action:
            if int(user_action) == ai_action:
                compare = "🤝 You matched AI!"
            else:
                compare = f"AI chose {ACTIONS[ai_action]}"

        fig_q = plot_q_values(q_values)
        fig_l = plot_learning()

        yield (
            decision,
            urgency.upper(),
            str(reward),
            f"{grade['reason']}\n{compare}",
            f"Based on sentiment={sentiment}",
            fig_q,
            fig_l
        )

    except Exception as e:
        yield ("Error", "Error", "Error", str(e), str(e), None, None)


# ⭐ Feedback function
def handle_feedback(rating):

    state = last_state["value"]
    action = last_action["value"]

    if state is None:
        return "⚠️ Run AI first!"

    if rating == 5:
        reward = 3
    elif rating == 4:
        reward = 2
    elif rating == 3:
        reward = 1
    elif rating == 2:
        reward = -1
    else:
        reward = -3

    alpha = 0.1
    old = q_table[state][action]
    new = old + alpha * (reward - old)

    q_table[state][action] = new

    reward_history.append(reward)

    np.save(QTABLE_FILE, q_table)
    np.save(REWARD_FILE, np.array(reward_history))

    return f"⭐ {rating}/5 → Reward {reward} | Updated Q: {round(new,2)}"


# 🎨 UI (MODERN GLASS DESIGN)
with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Segoe UI', sans-serif;
    }

    .glass {
        background: rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #00e5ff;
    }

    .subtitle {
        text-align: center;
        color: #ccc;
        margin-bottom: 20px;
    }

    button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
    }
    """
) as demo:

    gr.Markdown('<div class="title">🤖 Smart Support AI</div>')
    gr.Markdown('<div class="subtitle">RL-Based Decision System</div>')

    with gr.Row():

        with gr.Column():
            with gr.Group(elem_classes="glass"):
                gr.Markdown("### 🧾 Input")

                message = gr.Textbox(label="Message")

                intent = gr.Dropdown(
                    ["billing", "technical", "general", "complaint"],
                    label="Intent"
                )

                sentiment = gr.Dropdown(
                    ["positive", "neutral", "angry"],
                    label="Sentiment"
                )

                urgency = gr.Dropdown(
                    ["low", "medium", "high"],
                    label="Urgency"
                )

                user_action = gr.Dropdown(
                    ["0", "1", "2", "3"],
                    label="Your Guess"
                )

                btn = gr.Button("🚀 Run AI")

        with gr.Column():
            with gr.Group(elem_classes="glass"):
                gr.Markdown("### 🤖 Output")

                status = gr.Textbox(label="Decision")
                priority = gr.Textbox(label="Priority")
                reward_box = gr.Textbox(label="Reward")

                explanation = gr.Textbox(label="Explanation")
                reasoning = gr.Textbox(label="Reasoning")

    with gr.Row():
        with gr.Group(elem_classes="glass"):
            gr.Markdown("### 📊 Insights")

            q_chart = gr.Plot()
            learning_chart = gr.Plot()

    with gr.Row():
        with gr.Group(elem_classes="glass"):
            gr.Markdown("### ⭐ Feedback")

            rating = gr.Slider(1, 5, step=1, value=3)
            feedback_btn = gr.Button("Submit")
            feedback_output = gr.Textbox()

    btn.click(
        get_decision,
        inputs=[message, intent, sentiment, urgency, user_action],
        outputs=[status, priority, reward_box, explanation, reasoning, q_chart, learning_chart]
    )

    feedback_btn.click(
        handle_feedback,
        inputs=[rating],
        outputs=[feedback_output]
    )

demo.launch()