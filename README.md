<div align="center">
  
# 🤖 AGNN: Adaptive Goal Negotiation Network
**A Autonomous, Self-Healing Multi-Agent Framework for Complex Task Solving**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LM Studio Compatible](https://img.shields.io/badge/LM_Studio-Compatible-blueviolet.svg)](https://lmstudio.ai/)
![Local Only](https://img.shields.io/badge/Privacy-100%25_Local-brightgreen)

AGNN is an advanced research framework that coordinates small, local Large Language Models (LLMs) to solve complex engineering, architecture, and strategy problems *entirely autonomously*. By forcing models to negotiate roles, grade each other's work mathematically, and execute phased pipelines, AGNN achieves outputs previously reserved for massive frontier models.

</div>

---

## 🚀 The Pitch: Why AGNN?
Current AI agents are rigid. If a model hallucinates, the script breaks. If the user prompt changes drastically, the hardcoded agent roles fail to adapt. If memory limits are hit, the system crashes.

**AGNN takes a completely different approach.**
It is an orchestrated swarm that *self-adapts*, *self-corrects*, and *self-heals*. You give it one prompt—such as *"Design a scalable backend for a multiplayer game"*—and walk away. 

AGNN breaks the task down, negotiates who does what, filters out bad ideas using a mathematical scoring protocol, writes the drafts, reviews the work, and automatically recovers when the underlying hardware or models fail. It runs 100% offline, powered by local LLMs via LM Studio.

---

## 🧠 Core Architecture

AGNN operates on two distinct overarching tiers to guarantee quality and autonomous progression:

### **Tier-1: Dynamic Role Negotiation & Team Formation**
Before writing a single line of the final output, the models converse in a negotiation room. 
1. **Dynamic Specialization**: Instead of hardcoding "Frontend Agent" and "Backend Agent", the orchestrator reads the user prompt and generates *custom* roles dynamically (e.g., "Websocket Architect", "Database Schema Expert").
2. **The Bidding War**: Agents examine the roles and bid on them based on their internal system prompt capabilities. 
3. **Consensus Requirement**: If the agents cannot agree on a coherent plan (measured via a strict Consensus Score threshold), the system automatically triggers a re-negotiation round to resolve conflicts before executing.

### **Tier-2: Phase-Gated Execution Pipeline**
The actual work is divided into a rigid, 4-step pipeline to prevent models from rushing to the finish line:
* 🔬 **Research Phase**: Information gathering and constraint identification.
* 📊 **Analysis Phase**: Framework structuring and pattern matching.
* 📝 **Draft Phase**: Substantive content generation.
* 🕵️ **Review Phase**: Quality assurance and gap identification.

Transitions between phases are handled entirely autonomously by a `PhaseController` that detects semantic saturation—when agents stop generating novel ideas and start agreeing, the threshold is met, and the phase advances.

---

## 🛡️ Groundbreaking Autonomy Features

AGNN doesn't just manage the conversation; it actively manages *the hardware and the models themselves*.

#### 🔄 1. Live Model Hot-Swapping
*The problem*: If a model running locally crashes due to VRAM overflow or an HTTP timeout, traditional systems die.
*The AGNN Solution*: AGNN actively tracks API health and consecutive rejections. If an agent hits the failure limit, AGNN **dynamically unloads the model from RAM**, queries your LM Studio API for available models on disk, **loads a replacement**, un-blacklists the agent, and seamlessly injects it back into the conversation.

#### 🩹 2. Context Window Self-Healing
*The problem*: Small models (1B-3B parameters) easily hit token limits and throw `HTTP 400 Context Overflow` errors.
*The AGNN Solution*: Instead of crashing, the Orchestrator catches the 400 error, dynamically trims that specific agent's historical memory window (forcing it to "forget" the oldest messages), and retries. 

#### 🧮 3. Intelligent TIS Gating (Tier-0 Protocol)
Every single message an agent proposes is held in isolation and mathematically scored before being admitted to the conversation ledger. The **Tier-0 Interaction Score (TIS)** evaluates:
* **Semantic Distance**: Is this actually novel, or just restating the previous message?
* **Reciprocal Coherence**: Does this align with the task, or is it hallucinated rambling?
* **Information Gain (Entropy)**: Did this add substantive value to the document?

*If a message scores below the dynamic threshold, it is silently rejected, and the agent is forced to rewrite it or is bypassed entirely.*

#### 📐 4. Hardware-Aware Token Allocation
Instead of hardcoding max sizes, AGNN queries your local server's `/api/v1/models` endpoint to discover the exact `context_length` of the model currently loaded in RAM. It then mathematically allocates the maximum safe percentage of tokens for generation without causing an overflow.

---

## 📂 The Final Deliverable

At the end of an AGNN run, you aren't handed a messy chat transcript. 

A dedicated **Synthesizer Module** reviews the entire session's accepted ledger. It strips away all "Agent dialogue" and conversational boilerplate, identifies the highest-value contributions from the Draft and Review phases, and compiles them into a single, polished, professional Markdown document (`/agnn/outputs/`).

You provide a one-sentence prompt; 15 minutes later, you receive a clean, boardroom-ready deliverable.

---

## 🛠️ Usage

**Prerequisites:**
1. Python 3.10+
2. [LM Studio](https://lmstudio.ai/) running locally with the Server turned on (default: `http://localhost:1234`).
3. At least 2-3 local models downloaded (e.g., `gemma-3-1b`, `qwen2.5-coder-3b`).

**Run the CLI Autonomous Mode:**
```bash
# Start the autonomous loop
python -m agnn

# Enter your LM Studio Server URL (or press Enter for default)
# Enter your objective: "Write a 30-day GTM strategy for an AI note-taking app."
# Walk away.
```
