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

        self.task_type = task
        self.tasks = TASKS[task]
        self.index = 0

    def reset(self):
        self.index = 0
        return Observation(ticket_text=self.tasks[self.index]["text"])

    def state(self):
        return Observation(ticket_text=self.tasks[self.index]["text"])

    def step(self, action: Action):
        current = self.tasks[self.index]

        reward_value = 0.0

        # =========================
        # PRIORITY ONLY
        # =========================
        if self.task_type == "priority_only":
            if action.priority == current["priority"]:
                reward_value = 0.9
            elif action.priority:
                reward_value = 0.4
            else:
                reward_value = 0.1

        # =========================
        # CATEGORY ONLY
        # =========================
        elif self.task_type == "category_only":
            if action.category == current["category"]:
                reward_value = 0.9
            elif action.category:
                reward_value = 0.4
            else:
                reward_value = 0.1

        # =========================
        # FULL TRIAGE
        # =========================
        elif self.task_type == "full_triage":
            reward_value = 0.0

            # priority scoring
            if action.priority == current["priority"]:
                reward_value += 0.4
            elif action.priority:
                reward_value += 0.2

            # category scoring
            if action.category == current["category"]:
                reward_value += 0.4
            elif action.category:
                reward_value += 0.2

            # bonus for perfect match
            if (
                action.priority == current["priority"] and
                action.category == current["category"]
            ):
                reward_value += 0.1

        # =========================
        # CLAMP (VERY IMPORTANT)
        # =========================
        reward_value = max(0.0, min(1.0, reward_value))

        self.index += 1
        done = self.index >= len(self.tasks)

        next_obs = None if done else self.state()

        return next_obs, Reward(value=reward_value), done, {}