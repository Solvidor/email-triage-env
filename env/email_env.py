from pydantic import BaseModel


# ✅ Typed Models
class Observation(BaseModel):
    ticket_text: str


class Action(BaseModel):
    priority: str
    category: str


class Reward(BaseModel):
    value: float


# ✅ Tasks
TASKS = {
    "easy": [
        {"text": "Payment failed", "priority": "high", "category": "billing"},
        {"text": "App crash issue", "priority": "medium", "category": "technical"},
    ],
    "medium": [
        {"text": "Refund not processed yet", "priority": "high", "category": "billing"},
        {"text": "App is slow sometimes", "priority": "medium", "category": "technical"},
    ],
    "hard": [
        {"text": "Charged twice and app crashes", "priority": "high", "category": "billing"},
        {"text": "Login issue and payment failed", "priority": "high", "category": "technical"},
    ]
}


class EmailEnv:
    def __init__(self, task="easy"):
        self.tasks = TASKS[task]
        self.index = 0

    def reset(self):
        self.index = 0
        return Observation(ticket_text=self.tasks[self.index]["text"])

    def state(self):
        return Observation(ticket_text=self.tasks[self.index]["text"])

    def step(self, action: Action):
        correct = self.tasks[self.index]

        reward_value = 0.0

        if action.priority == correct["priority"]:
            reward_value += 0.5
        if action.category == correct["category"]:
            reward_value += 0.5

        if reward_value == 0.0:
            reward_value = -0.2

        self.index += 1
        done = self.index >= len(self.tasks)

        next_obs = None if done else self.state()

        return next_obs, Reward(value=reward_value), done, {}