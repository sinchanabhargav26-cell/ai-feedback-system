import numpy as np

# Dummy environment state
state = 0

def reset():
    global state
    state = 0
    return {"state": state}

def step(action):
    global state

    # simple logic (dummy)
    reward = 1 if action == state else -1
    state = (state + 1) % 5

    done = False

    return {
        "state": state,
        "reward": reward,
        "done": done
    }
