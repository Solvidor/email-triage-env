from fastapi import FastAPI
from pydantic import BaseModel
from env.email_env import EmailEnv, Action

app = FastAPI()

# Default task (can change via reset)
env = EmailEnv(task="priority_only")


class ActionRequest(BaseModel):
    priority: str = None
    category: str = None


class ResetRequest(BaseModel):
    task: str = "priority_only"


# 🔄 RESET ENDPOINT
@app.post("/reset")
def reset(req: ResetRequest):
    global env
    env = EmailEnv(task=req.task)
    state = env.reset()
    return state.model_dump()


# ▶ STEP ENDPOINT
@app.post("/step")
def step(action: ActionRequest):
    action_obj = Action(**action.model_dump())
    state, reward, done, info = env.step(action_obj)

    return {
        "observation": state.model_dump() if state else None,
        "reward": reward.value,
        "done": done,
        "info": info
    }


# 🔍 STATE ENDPOINT
@app.get("/state")
def get_state():
    state = env.state()
    return state.model_dump()
import uvicorn

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()