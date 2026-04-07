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

TASK = "easy"
BENCHMARK = "email-triage-env"


# ==============================
# OPTIONAL OPENAI CLIENT
# ==============================
client = None
if API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )


# ==============================
# LOGGING FUNCTIONS
# ==============================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str):
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[str]):
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={','.join(rewards)}",
        flush=True
    )


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    env = EmailEnv(task=TASK)
    state = env.reset()

    log_start(TASK, BENCHMARK, MODEL_NAME)

    step_num = 0
    rewards = []
    done = False

    try:
        while not done:
            step_num += 1
            error = "null"

            # ===== LLM RESPONSE (SAFE) =====
            response = ""

            if client:
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": "Classify support ticket into priority (high/medium/low) and category (billing/technical/general). Return only category name."
                            },
                            {
                                "role": "user",
                                "content": state.ticket_text
                            }
                        ],
                        temperature=0
                    )

                    response = (completion.choices[0].message.content or "").lower()

                except Exception as e:
                    error = str(e)

            text = state.ticket_text.lower()

            # ===== CATEGORY =====
            if response:
                if "billing" in response:
                    category = "billing"
                elif "technical" in response:
                    category = "technical"
                else:
                    category = "general"
            else:
                if "payment" in text or "refund" in text:
                    category = "billing"
                elif "crash" in text or "bug" in text:
                    category = "technical"
                else:
                    category = "general"

            # ===== IMPROVED PRIORITY LOGIC =====
            if "urgent" in text or "asap" in text or "immediately" in text:
                priority = "high"
            elif "slow" in text or "delay" in text:
                priority = "medium"
            else:
                priority = "high"

            action = Action(priority=priority, category=category)

            state, reward, done, info = env.step(action)

            rewards.append(f"{reward.value:.2f}")

            log_step(
                step=step_num,
                action=str(action.model_dump()),
                reward=reward.value,
                done=done,
                error=error
            )

        # ===== SCORE =====
        numeric_rewards = [float(r) for r in rewards]
        score = sum(numeric_rewards) / len(numeric_rewards) if rewards else 0.0

        success = done

        log_end(success, step_num, score, rewards)

    except Exception:
        log_end(False, step_num, 0.0, rewards)


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()