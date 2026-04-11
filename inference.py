import os
from typing import List
from openai import OpenAI
from env.email_env import EmailEnv, Action


# ==============================
# ENV VARIABLES
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "email-triage-env"


# ==============================
# OPENAI CLIENT (OPTIONAL)
# ==============================
client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ==============================
# SAFE CLAMP FUNCTION (CRITICAL)
# ==============================
def safe_value(x: float) -> float:
    return max(0.0001, min(0.9999, x))


# ==============================
# LOGGING FUNCTIONS (STRICT FORMAT)
# ==============================
def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str):
    reward = safe_value(reward)  # 🔥 ensure safe before logging
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error}",
        flush=True
    )


def log_end(success: bool, steps: int, rewards: List[str]):
    # compute score from rewards
    numeric_rewards = [float(r) for r in rewards] if rewards else [0.5]
    score = sum(numeric_rewards) / len(numeric_rewards)

    # strict clamp (VERY IMPORTANT)
    score = max(0.0001, min(0.9999, score))

    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={','.join(rewards)}",
        flush=True
    )


# ==============================
# ACTION LOGIC (LLM + FALLBACK)
# ==============================
def get_action(state_text: str):
    text = state_text.lower()

    # 🔥 Try OpenAI API first
    if client:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Classify support ticket into priority (high/medium/low) and category (billing/technical/general). Return format: priority,category"
                    },
                    {
                        "role": "user",
                        "content": state_text
                    }
                ],
                temperature=0
            )

            response = (completion.choices[0].message.content or "").lower()

            parts = response.split(",")
            if len(parts) == 2:
                return Action(
                    priority=parts[0].strip(),
                    category=parts[1].strip()
                )

        except Exception:
            pass  # fallback if API fails

    # ✅ SAFE FALLBACK (deterministic)
    if "payment" in text or "refund" in text:
        category = "billing"
    elif "crash" in text or "bug" in text:
        category = "technical"
    else:
        category = "general"

    if "urgent" in text or "asap" in text:
        priority = "high"
    elif "slow" in text or "delay" in text:
        priority = "medium"
    else:
        priority = "high"

    return Action(priority=priority, category=category)


# ==============================
# RUN SINGLE TASK
# ==============================
def run_task(task_name: str):
    env = EmailEnv(task=task_name)
    state = env.reset()

    log_start(task_name)

    step_num = 0
    rewards = []
    done = False

    try:
        while not done:
            step_num += 1
            error = "null"

            action = get_action(state.ticket_text)

            state, reward, done, info = env.step(action)

            # 🔥 Clamp reward BEFORE storing/logging
            safe_reward = safe_value(reward.value)

            rewards.append(f"{safe_reward:.2f}")

            log_step(
                step=step_num,
                action=str(action.model_dump()),
                reward=safe_reward,
                done=done,
                error=error
            )

        log_end(True, step_num, rewards)

    except Exception:
        log_end(False, step_num, rewards)


# ==============================
# MAIN
# ==============================
def main():
    tasks = ["priority_only", "category_only", "full_triage"]

    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()