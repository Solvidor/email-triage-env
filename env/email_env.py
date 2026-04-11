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
# TASKS
# =========================
TASKS = {
    "priority_only": [
        {
            "text": "URGENT: Payment failed and customer is furious",
            "priority": "high"
        },
        {
            "text": "App occasionally crashes after update",
            "priority": "medium"
        }
    ],

    "category_only": [
        {
            "text": "Refund not processed after 5 days, very frustrated",
            "category": "billing"
        },
        {
            "text": "App UI freezes when clicking button",
            "category": "technical"
        }
    ],

    "full_triage": [
        {
            "text": "Customer extremely angry: charged twice and app crashes",
            "priority": "high",
            "category": "billing"
        },
        {
            "text": "User reports login issue and payment failure",
            "priority": "high",
            "category": "technical"
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

        # =========================
        # UNIFIED GRADER (FINAL)
        # =========================
        reward_value = 0.3  # base

        # PRIORITY (always evaluated)
        if action.priority == current.get("priority"):
            reward_value += 0.2
        elif action.priority is not None:
            reward_value += 0.1
        else:
            reward_value += 0.05

        # CATEGORY (always evaluated)
        if action.category == current.get("category"):
            reward_value += 0.2
        elif action.category is not None:
            reward_value += 0.1
        else:
            reward_value += 0.05

        # SMALL STEP VARIATION
        reward_value += (self.index * 0.01)

        # =========================
        # STRICT RANGE (CRITICAL)
        # =========================
        reward_value = max(0.05, min(0.95, reward_value))
        reward_value = round(reward_value, 4)

        self.index += 1
        done = self.index >= len(self.tasks)

        next_obs = None if done else self.state()

        return next_obs, Reward(value=reward_value), done, {}