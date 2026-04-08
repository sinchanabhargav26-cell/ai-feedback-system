"""
env/__init__.py
===============
Package-level exports for the Customer Support RL Environment.

Import from here for the cleanest usage:
    from env import CustomerSupportEnv, Grader, Task, ACTION_LABELS
"""

from env.support_env import CustomerSupportEnv
from env.task import (
    TaskLoader, Task, ACTION_LABELS, ACTION_LABELS_SHORT,
    INTENT_MAP, SENTIMENT_MAP, URGENCY_MAP,
    INTENT_INV, SENTIMENT_INV, URGENCY_INV,
    VALID_INTENTS, VALID_SENTIMENTS, VALID_URGENCIES,
    OBS_LOW, OBS_HIGH,
)
from env.grader import (
    Grader, GradeResult,
    ACTION_AUTO_RESPOND, ACTION_CLARIFY, ACTION_ESCALATE, ACTION_FALLBACK,
    REWARD_CORRECT_ACTION, REWARD_CORRECT_ESCALATION,
    REWARD_GOOD_CLARIFICATION, REWARD_INCORRECT_ACTION,
    REWARD_UNNECESSARY_ESCALATION, REWARD_OVERUSE_FALLBACK,
    REWARD_FAILED_ESCALATION, ESCALATION_COST_PENALTY,
)

__all__ = [
    # Environment
    "CustomerSupportEnv",
    # Task
    "TaskLoader", "Task",
    # Maps & Labels
    "ACTION_LABELS", "ACTION_LABELS_SHORT",
    "INTENT_MAP", "SENTIMENT_MAP", "URGENCY_MAP",
    "INTENT_INV", "SENTIMENT_INV", "URGENCY_INV",
    "VALID_INTENTS", "VALID_SENTIMENTS", "VALID_URGENCIES",
    "OBS_LOW", "OBS_HIGH",
    # Grader
    "Grader", "GradeResult",
    # Action constants
    "ACTION_AUTO_RESPOND", "ACTION_CLARIFY", "ACTION_ESCALATE", "ACTION_FALLBACK",
    # Reward constants
    "REWARD_CORRECT_ACTION", "REWARD_CORRECT_ESCALATION",
    "REWARD_GOOD_CLARIFICATION", "REWARD_INCORRECT_ACTION",
    "REWARD_UNNECESSARY_ESCALATION", "REWARD_OVERUSE_FALLBACK",
    "REWARD_FAILED_ESCALATION", "ESCALATION_COST_PENALTY",
]
