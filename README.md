# Student Tutor

A stateful AI tutoring chatbot built with LangGraph that helps students study for UNEB UCE and UACE exams. The tutor remembers conversation history across sessions, adapts to the student's confidence level, generates quizzes, evaluates answers, and escalates to a human teacher when a student is genuinely struggling.

---

## What it does

- Explains concepts clearly using simple language and local Ugandan examples
- Tracks the student's current topic and confidence level across sessions
- Generates short quizzes and evaluates student answers
- Routes to an escalation node when confidence drops below 4/10
- Persists full conversation state to SQLite — sessions survive server restarts
- Supports any UNEB subject — student picks subject at start and asks freely
- Clean chat UI with live confidence and topic indicators

---

## How it works

The conversation flow is modelled as a graph. Each node is a processing step; edges are transitions between them. The graph decides which node to run next based on the current state.

```
         ┌─────────────────────────────┐
         ▼                             │
     [tutor_node]                    loop
         │                             │
    ─────┼─────────────────────────    │
    │              │          │        │
  quiz?        escalate?    done      │
    │              │          │        │
    ▼              ▼          ▼        │
[quiz_node]  [escalate_node]  END      │
    │                                  │
    ▼                                  │
[evaluate_node] ──────────────────────┘
    │
  confidence < 4?
    │
    ▼
[escalate_node]
```

### The nodes

**tutor_node** — main teaching node. Explains concepts, answers questions, and decides whether to offer a quiz or escalate based on the student's messages.

**quiz_node** — generates 3 UNEB-style questions on the current topic: one recall, one application, one explanation.

**evaluate_node** — reads the student's quiz answers, gives feedback, awards correct answers, and updates the confidence score (0–10).

**escalate_node** — triggered when confidence drops below 4 or the student signals they are lost. Sends an encouraging message and logs the escalation for teacher follow-up.

---

## Project structure

```
student-tutor/
├── graph.py           # LangGraph state, nodes, edges, graph assembly
├── main.py            # FastAPI app — chat endpoint + session management
├── templates/
│   └── index.html     # Chat UI with subject picker and status indicators
├── memory.db          # SQLite checkpoint store (auto-created on first run)
├── .env               # Environment variables (never commit this)
├── .env.example       # Template for required variables
└── README.md
```

---

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4o)

---

## Installation

```bash
# 1. Navigate to project folder
cd student-tutor

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install langgraph langchain-openai langchain-core \
            fastapi uvicorn python-multipart jinja2 \
            python-dotenv
```

---

## Configuration

Copy `.env.example` to `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
```

---

## Running the app

```bash
source venv/bin/activate
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Usage

1. Enter your name and pick a subject from the dropdown
2. Click **Start learning** — the tutor greets you and asks what you want to study
3. Ask about any topic within your subject
4. Type **"give me a quiz"** or **"quiz me"** to trigger the quiz node
5. Answer the quiz questions — the tutor evaluates and updates your confidence score
6. Say **"I don't understand"** or **"I'm lost"** to trigger teacher escalation

### Available subjects

Biology, Chemistry, Physics, Mathematics, History, Geography, English Language, Economics, Computer Studies

---

## Session persistence

Every conversation turn is saved to `memory.db` (SQLite). The session ID generated at the start ties all turns together.

**What this means in practice:**

- Close the browser and come back — the tutor remembers exactly where you left off
- Restart the server — sessions are still there
- Same student, different device — use the same session ID and history is restored

The session ID is shown in the header. Keep it to resume a session later.

```python
# How state is saved — handled automatically by LangGraph's SqliteSaver
memory = SqliteSaver.from_conn_string("memory.db")
graph  = builder.compile(checkpointer=memory)

# Every invoke call with the same thread_id restores and saves state
config = {"configurable": {"thread_id": session_id}}
result = graph.invoke(input_state, config=config)
```

---

## State schema

Everything the graph carries between turns:

```python
class TutorState(TypedDict):
    messages:     list        # full conversation history (append-only)
    subject:      str         # e.g. "Biology"
    topic:        str         # current topic being studied
    confidence:   int         # 0–10, updated after each quiz
    quiz_pending: bool        # True when a quiz was just issued
    escalated:    bool        # True if handed to human teacher
    student_name: str         # used to personalise responses
```

---

## Routing logic

The graph uses conditional edges to decide what happens after each node:

```python
# After tutor_node
if escalated   → escalate_node
if quiz_pending → quiz_node
else            → END  (wait for next student message)

# After evaluate_node
if confidence < 4 or escalated → escalate_node
else                            → tutor_node
```

The student's confidence score is the primary decision signal. A score below 4 means the student is seriously struggling and needs human intervention.

---

## Confidence score guide

| Score | Meaning | Action |
|-------|---------|--------|
| 8–10 | Strong understanding | Tutor suggests moving to next topic |
| 5–7 | Partial understanding | Tutor offers more examples and practice |
| 3–4 | Struggling | Tutor slows down, re-explains from basics |
| 0–2 | Seriously lost | Escalate to human teacher |

---

## Teacher escalation

When escalation is triggered, the following is logged to the server terminal:

```
[ESCALATION] Student: Sarah | Subject: Biology | Topic: Photosynthesis | Confidence: 2/10
```

In production, replace the `print` statement in `escalate_node` with your preferred notification method:

```python
# Email
send_email(teacher_email, subject=f"Student needs help: {state['topic']}", ...)

# SMS via Africa's Talking
africastalking.SMS.send(teacher_phone, message=...)

# Slack / Teams webhook
httpx.post(webhook_url, json={"text": f"Student {state['student_name']} needs help..."})

# Laravel / GrantFlow API
httpx.post(f"{BACKEND_URL}/api/escalations/", json={...})
```

---

## Key LangGraph concepts used

**TypedDict state** — a typed dictionary that flows through every node. Each node receives the full state and returns only the fields it changed. LangGraph merges the partial update automatically.

**add_messages reducer** — the `Annotated[list, add_messages]` annotation on the messages field tells LangGraph to append new messages rather than replace the entire list. This is how conversation history accumulates across turns.

**Conditional edges** — routing functions that read the current state and return a string indicating which node to run next. The graph cannot go somewhere you have not wired up — routing is always explicit.

**SqliteSaver checkpointer** — serialises the full state to SQLite after every turn and restores it at the start of the next. Identified by `thread_id` — one thread per student session.

**Entry point** — `builder.set_entry_point("tutor")` means every new conversation always starts at the tutor node regardless of which node ended the previous turn.

---

## Extending the tutor

### Add a new subject

No code changes needed — the subject is passed as a string in the UI and injected into the system prompt dynamically.

### Add a document/notes tool

Give the tutor access to the student's own notes:

```python
@tool
def search_student_notes(query: str) -> str:
    """Search the student's uploaded notes for relevant content."""
    # implement vector search over uploaded PDFs
    ...

tutor_agent = Agent(tools=[search_student_notes], ...)
```

### Add a progress dashboard

Query `memory.db` directly to show a student's confidence history over time:

```python
import sqlite3

conn = sqlite3.connect("memory.db")
# LangGraph stores checkpoints in the 'checkpoints' table
# Query by thread_id to get session history
```

### Integrate with lav_sms (school management system)

Pull the student's class, stream, and subjects from your Laravel school management system and pre-populate the session:

```python
student = httpx.get(f"{LAV_SMS_URL}/api/students/{student_id}").json()
subject = student["current_subjects"][0]
```

---

## Performance notes

- Response time: **3–8 seconds** per turn (one GPT-4o call per node)
- Quiz evaluation: **5–12 seconds** (more complex prompt)
- SQLite handles hundreds of concurrent sessions without configuration
- For production with many simultaneous users, switch to PostgreSQL checkpointer:

```python
from langgraph.checkpoint.postgres import PostgresSaver
memory = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")
```

---

## Difference from a simple chatbot

A regular chatbot has no memory between page refreshes and no structured flow. This tutor:

| Feature | Simple chatbot | This tutor |
|---------|---------------|------------|
| Memory across sessions | ✗ | ✓ SQLite persistence |
| Structured quiz flow | ✗ | ✓ Dedicated quiz + evaluate nodes |
| Confidence tracking | ✗ | ✓ Updated after every quiz |
| Teacher escalation | ✗ | ✓ Automatic when confidence < 4 |
| Conditional routing | ✗ | ✓ Graph branches on state |
| Topic tracking | ✗ | ✓ Updates when student switches topics |

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Agent framework | LangGraph |
| LLM | OpenAI GPT-4o via LangChain |
| State persistence | SQLite (SqliteSaver checkpointer) |
| Web framework | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

---
