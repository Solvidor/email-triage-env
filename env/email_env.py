from pydantic import BaseModel


# ✅ Models
class Observation(BaseModel):
    ticket_text: str


class Action(BaseModel):
    priority: str = None
    category: str = None


class Reward(BaseModel):
    value: float


# ✅ TASKS (with sentiment - creative twist)
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


# ✅ ENV CLASS
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

        # ✅ DIFFERENT GRADERS
        if self.task_type == "priority_only":
            if action.priority == current["priority"]:
                reward_value = 1.0
            else:
                reward_value = 0.0

        elif self.task_type == "category_only":
            if action.category == current["category"]:
                reward_value = 1.0
            else:
                reward_value = 0.0

        elif self.task_type == "full_triage":
            if action.priority == current["priority"]:
                reward_value += 0.4
            else:
                reward_value -= 0.2

            if action.category == current["category"]:
                reward_value += 0.4
            else:
                reward_value -= 0.2

            # ✅ bonus for perfect match
            if (
                action.priority == current["priority"] and
                action.category == current["category"]
            ):
                reward_value += 0.2

        # ✅ penalty for empty action
        if not action.priority and not action.category:
            reward_value -= 0.5

        # 🔥 CREATIVE TWIST: sentiment-based reward shaping
        sentiment = current.get("sentiment", "neutral")

        if sentiment == "angry":
            reward_value *= 1.2   # higher stakes

        elif sentiment == "calm":
            reward_value *= 0.9   # less critical

        self.index += 1
        done = self.index >= len(self.tasks)

        next_obs = None if done else self.state()

        return next_obs, Reward(value=reward_value), done, {}