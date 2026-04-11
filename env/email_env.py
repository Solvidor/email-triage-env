from pydantic import BaseModel


# =========================
# MODELS
# =========================
class Observation(BaseModel):
    ticket_text: str


class Action(BaseModel):
    priority: str = None
    category: str = None


class Reward(BaseModel):
    value: float


# =========================
# TASKS (with sentiment - creative twist)
# =========================
TASKS = {
    "priority_only": [
        {
            "text": "URGENT: Payment failed and customer is furious",
            "priority": "high",
            "sentiment": "angry"
        },
        {
            "text": "App occasionally crashes after update",
            "priority": "medium",
            "sentiment": "neutral"
        }
    ],

    "category_only": [
        {
            "text": "Refund not processed after 5 days, very frustrated",
            "category": "billing",
            "sentiment": "angry"
        },
        {
            "text": "App UI freezes when clicking button",
            "category": "technical",
            "sentiment": "neutral"
        }
    ],

    "full_triage": [
        {
            "text": "Customer extremely angry: charged twice and app crashes",
            "priority": "high",
            "category": "billing",
            "sentiment": "angry"
        },
        {
            "text": "User politely reports login issue and payment failure",
            "priority": "high",
            "category": "technical",
            "sentiment": "calm"
        }
    ]
}


# =========================
# ENV CLASS
# =========================
class EmailEnv:
    def __init__(self, task="priority_only"):
        if task not in TASKS:
            raise ValueError("Invalid task type")

        self.tasks = TASKS[task]
        self.index = 0

    def reset(self):
        self.index = 0
        return Observation(ticket_text=self.tasks[self.index]["text"])

    def state(self):
        return Observation(ticket_text=self.tasks[self.index]["text"])

    def step(self, action: Action):
        current = self.tasks[self.index]

        reward_value = 0.3  # base

    # PRIORITY
        if "priority" in current:
           if action.priority == current["priority"]:
                reward_value += 0.25
           else:
                reward_value += 0.05

    # CATEGORY
        if "category" in current:
           if action.category == current["category"]:
                reward_value += 0.25
           else:
                reward_value += 0.05

    # ADD VARIATION (CRITICAL FIX)
        reward_value += (self.index * 0.02)

    # STRICT RANGE
        reward_value = max(0.05, min(0.95, reward_value))
        reward_value = round(reward_value, 4)

        self.index += 1
        done = self.index >= len(self.tasks)

        next_obs = None if done else self.state()

        return next_obs, Reward(value=reward_value), done, {}