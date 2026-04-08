def grade_action(action, urgency):
    if urgency == "high":
        if action == 2:
            return {
                "correct": True,
                "reward": 3,
                "reason": "Correct escalation for high urgency"
            }
        else:
            return {
                "correct": False,
                "reward": -5,
                "reason": "Failed to escalate critical issue"
            }

    if action == 0:
        return {
            "correct": True,
            "reward": 2,
            "reason": "Good auto-response"
        }

    if action == 1:
        return {
            "correct": True,
            "reward": 1,
            "reason": "Clarification is acceptable"
        }

    return {
        "correct": False,
        "reward": -1,
        "reason": "Fallback is suboptimal"
    }