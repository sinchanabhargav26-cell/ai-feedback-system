import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ✅ FIXED: All mappings defined
SENTIMENT_MAP = {"positive": 0, "neutral": 1, "angry": 2}
INTENT_MAP = {"billing": 0, "technical": 1, "general": 2, "complaint": 3}
URGENCY_MAP = {"low": 0, "medium": 1, "high": 2}


class SupportEnv(gym.Env):
    def __init__(self):
        super(SupportEnv, self).__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(3,),
            dtype=np.float32
        )

        self.state = None

    def encode_state(self, intent, sentiment, urgency):
        return np.array([
            INTENT_MAP[intent],
            SENTIMENT_MAP[sentiment],
            URGENCY_MAP[urgency]
        ], dtype=np.float32)

    def reset(self, intent="general", sentiment="neutral", urgency="low"):
        self.state = self.encode_state(intent, sentiment, urgency)
        return self.state, {}

    def step(self, action):
        intent, sentiment, urgency = self.state

        reward = 0

        # ✅ Improved reward logic
        if urgency == 2:  # high urgency
            if action == 2:
                reward = 3
            else:
                reward = -5
        elif action == 0:
            reward = 2
        elif action == 1:
            reward = 1
        elif action == 3:
            reward = -1

        return self.state, reward, True, False, {}