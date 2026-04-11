title: Email Triage OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
# 📧 Sentiment-Aware Customer Support Triage Environment

## 🚀 Overview

This project implements a **real-world OpenEnv environment** simulating customer support ticket triage.
An AI agent must classify incoming support requests based on:

* **Priority** (high / medium / low)
* **Category** (billing / technical / general)

What makes this environment unique is the introduction of **sentiment-aware reward shaping**, where emotionally charged tickets influence the reward signal.

---

## 🧠 Motivation

In real-world customer support systems:

* Angry customers require faster and more accurate responses
* Misclassification of urgent issues leads to poor user experience

This environment models that behavior by:

* Introducing **sentiment-based urgency**
* Penalizing incorrect decisions more heavily for critical cases
* Rewarding precise handling of high-stakes scenarios

---

## ⚙️ Environment Design

### Observation Space

```json
{
  "ticket_text": "string"
}
```

### Action Space

```json
{
  "priority": "high | medium | low",
  "category": "billing | technical | general"
}
```

### Reward Function

The reward is designed to provide **dense feedback**:

* Correct prediction → positive reward
* Partial correctness → partial reward
* Incorrect prediction → penalty
* Empty/invalid action → strong penalty

Additionally:

* **Angry tickets → amplified reward/penalty (×1.2)**
* **Calm tickets → reduced impact (×0.9)**

---

## 🧩 Tasks

The environment contains **three distinct tasks with independent graders**:

### 1. Priority Classification

* Agent predicts urgency level
* Evaluates ability to detect critical issues

### 2. Category Classification

* Agent identifies issue type (billing / technical)
* Tests semantic understanding

### 3. Full Triage (Hard Task)

* Agent predicts both priority and category
* Includes **multi-objective grading + sentiment impact**

---

## 📊 Reward Strategy

| Scenario          | Reward            |
| ----------------- | ----------------- |
| Fully correct     | High reward       |
| Partially correct | Partial reward    |
| Incorrect         | Penalty           |
| Empty action      | Strong penalty    |
| Angry ticket      | Amplified outcome |
| Calm ticket       | Reduced impact    |

---

## 🤖 Baseline Behavior

A baseline agent can:

* Read ticket text
* Predict labels using an LLM
* Achieve reproducible scores across tasks

---

## 🐳 Running the Environment

### Build Docker Image

```bash
docker build -t openenv-project .
```

### Run Container

```bash
docker run -p 7860:7860 openenv-project
```

### Test API

```bash
curl -X POST http://localhost:7860/reset
```

---

## 📦 Project Structure

```
openenv-project/
│── env/
│── server/
│── openenv.yaml
│── pyproject.toml
│── Dockerfile
│── inference.py
```

---

## 🌟 Key Features

* ✅ Real-world customer support simulation
* ✅ Multi-task evaluation with independent graders
* ✅ Sentiment-aware reward shaping (novel contribution)
* ✅ Deterministic scoring system
* ✅ OpenEnv compliant & Dockerized

---

## 🏁 Conclusion

This environment provides a **realistic and challenging benchmark** for evaluating AI agents in customer support scenarios.

By incorporating **human emotion into reward dynamics**, it introduces a layer of complexity that better reflects real-world decision-making.

---

## 🔮 Future Improvements

* Multi-turn conversations
* Context-aware ticket history
* Escalation workflows
* Response generation evaluation

---
