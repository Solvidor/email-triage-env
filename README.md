---
title: Email Triage OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---
# Email Triage OpenEnv

## Description

This environment simulates a real-world customer support workflow where an AI agent must classify incoming support tickets into the correct priority and category.

It is designed for training and evaluating agents on structured decision-making tasks.

---

## Observation Space

The agent receives:

* `ticket_text` (string): The content of the customer support request.

Example:

```json
{
  "ticket_text": "Payment failed while placing order"
}
```

---

## Action Space

The agent must output:

* `priority`: `"high"`, `"medium"`, `"low"`
* `category`: `"billing"`, `"technical"`, `"general"`

Example:

```json
{
  "priority": "high",
  "category": "billing"
}
```

---

## Tasks

### Easy

* Simple, single-issue tickets
* Clear mapping to category and priority

### Medium

* Slightly ambiguous tickets
* Requires better understanding of context

### Hard

* Multi-issue or confusing tickets
* Requires reasoning across multiple signals

---

## Reward Function

* **1.0** → Both priority and category correct
* **0.5** → One correct (partial match)
* **-0.2** → Both incorrect

The reward provides meaningful feedback across the trajectory.

---

## Environment API

### reset()

Returns initial observation

### step(action)

Returns:

* observation
* reward
* done
* info

### state()

Returns current state

---

## Setup Instructions

Build the Docker container:

```bash
docker build -t openenv-project .
```

Run the environment:

```bash
docker run -p 7860:7860 openenv-project
```

---

## Usage

### Reset environment

```bash
POST /reset
```

### Take a step

```bash
POST /step
```

---

## Baseline Inference

Run:

```bash
python inference.py
```

This script:

* Executes the environment
* Logs output in required `[START]`, `[STEP]`, `[END]` format
* Produces reproducible baseline scores

---

## Deployment

This environment is containerized and deployed using:

* Docker
* Hugging Face Spaces (Docker SDK)

---

## Notes

* Fully compliant with OpenEnv specification
* Includes typed models using Pydantic
* Supports deterministic evaluation with reproducible results
