from fastapi import FastAPI
from env.email_env import EmailEnv, Action
import uvicorn

app = FastAPI()
env = EmailEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": obs.dict() if obs else None,
        "reward": reward.value,
        "done": done,
        "info": info
    }

# ✅ REQUIRED for OpenEnv
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

# ✅ REQUIRED check
if __name__ == "__main__":
    main()