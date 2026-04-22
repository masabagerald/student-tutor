import os
import json
from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
#from langgraph.checkpoint import SqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# ── State ───────────────────────────────────────────────────────
# Everything the graph carries between turns lives here.
# add_messages is a reducer — it appends new messages rather than
# replacing the whole list, so history accumulates automatically.

class TutorState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    subject: str                              # e.g. "Biology"
    topic: str                                # current topic being studied
    confidence: int                           # 0-10, updated after each quiz
    quiz_pending: bool                        # True when a quiz was just given
    escalated: bool                           # True if handed to human teacher
    student_name: str                         # personalises responses


# ── Nodes ───────────────────────────────────────────────────────

def tutor_node(state: TutorState) -> dict:
    """Main teaching node — explains concepts, answers questions."""

    system = SystemMessage(content=f"""
You are a patient, encouraging UNEB tutor helping {state['student_name']} 
study {state['subject']}.

Current topic: {state['topic'] or 'not yet set — ask the student what they want to study'}
Student confidence level: {state['confidence']}/10

Your behaviour:
- Explain concepts clearly using simple language and local Ugandan examples where helpful.
- After explaining a concept, offer to give the student a short quiz to test understanding.
- If the student asks for a quiz directly, set quiz_mode to true in your response.
- If the student seems confused (says things like "I don't understand", "I'm lost", 
  "this is too hard"), set escalate to true.
- Keep responses focused and encouraging. Never make the student feel stupid.

Respond in this JSON format:
{{
  "reply": "your message to the student",
  "topic": "current topic (update if student changed it)",
  "quiz_mode": false,
  "escalate": false
}}
""")

    response = llm.invoke([system] + state["messages"])

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        data = {
            "reply": response.content,
            "topic": state["topic"],
            "quiz_mode": False,
            "escalate": False,
        }

    updates: dict = {
        "messages": [AIMessage(content=data["reply"])],
        "topic": data.get("topic", state["topic"]),
        "quiz_pending": False,
        "escalated": False,
    }

    if data.get("quiz_mode"):
        updates["quiz_pending"] = True
    if data.get("escalate"):
        updates["escalated"] = True

    return updates


def quiz_node(state: TutorState) -> dict:
    """Generates a short quiz on the current topic."""

    system = SystemMessage(content=f"""
You are a UNEB examiner creating a short quiz for {state['student_name']} 
on {state['subject']} — specifically {state['topic']}.

Write exactly 3 questions appropriate for UNEB level:
- 1 recall question (definition or fact)
- 1 application question  
- 1 short explanation question

Number them clearly. Tell the student to answer all three.
End with: "Take your time — answer when you're ready!"

Respond with just the quiz text, no JSON needed.
""")

    response = llm.invoke([system] + state["messages"])

    return {
        "messages": [AIMessage(content=response.content)],
        "quiz_pending": True,
    }


def evaluate_node(state: TutorState) -> dict:
    """Evaluates the student's quiz answers and updates confidence."""

    system = SystemMessage(content=f"""
You are a UNEB tutor evaluating {state['student_name']}'s quiz answers 
on {state['subject']} — {state['topic']}.

Review the conversation above. The most recent student message contains their answers.

Evaluate each answer, give correct answers for any they got wrong, 
and provide an encouraging summary.

Then estimate the student's confidence/understanding as a score from 0-10.
- 0-3: seriously struggling, needs teacher escalation
- 4-6: partial understanding, needs more practice  
- 7-10: good understanding, ready to move on

Respond in this JSON format:
{{
  "reply": "your evaluation and feedback message",
  "confidence": 7,
  "escalate": false
}}

Set escalate to true only if confidence is below 4.
""")

    response = llm.invoke([system] + state["messages"])

    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        data = {
            "reply": response.content,
            "confidence": state["confidence"],
            "escalate": False,
        }

    return {
        "messages": [AIMessage(content=data["reply"])],
        "confidence": data.get("confidence", state["confidence"]),
        "quiz_pending": False,
        "escalated": data.get("escalate", False),
    }


def escalate_node(state: TutorState) -> dict:
    """Notifies a human teacher when the student is really struggling."""

    message = (
        f"I can see {state['topic']} in {state['subject']} is genuinely challenging "
        f"for you right now, and that's completely okay. I've flagged this for your "
        f"teacher so they can give you extra support. In the meantime, let's slow "
        f"down — is there one specific part you'd like me to explain again from scratch?"
    )

    # In production: send email/SMS to teacher here
    print(f"[ESCALATION] Student: {state['student_name']} | "
          f"Subject: {state['subject']} | Topic: {state['topic']} | "
          f"Confidence: {state['confidence']}/10")

    return {
        "messages": [AIMessage(content=message)],
        "escalated": True,
    }


# ── Routing logic ───────────────────────────────────────────────

def route_after_tutor(state: TutorState) -> str:
    """Decides where to go after the tutor node responds."""
    if state.get("escalated"):
        return "escalate"
    if state.get("quiz_pending"):
        return "quiz"
    return END


def route_after_evaluate(state: TutorState) -> str:
    """Decides where to go after quiz evaluation."""
    if state.get("escalated") or state.get("confidence", 10) < 4:
        return "escalate"
    return "tutor"


# ── Graph assembly ──────────────────────────────────────────────

def build_graph():
    builder = StateGraph(TutorState)

    # Add nodes
    builder.add_node("tutor",    tutor_node)
    builder.add_node("quiz",     quiz_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("escalate", escalate_node)

    # Entry point
    builder.set_entry_point("tutor")

    # Conditional edge after tutor — branch on state
    builder.add_conditional_edges(
        "tutor",
        route_after_tutor,
        {
            "quiz":     "quiz",
            "escalate": "escalate",
            END:        END,
        }
    )

    # Quiz always leads to evaluate
    builder.add_edge("quiz", END)        # quiz output goes to student first
    builder.add_edge("escalate", END)    # escalation message ends the turn

    # After evaluate, route back to tutor or escalate
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "tutor":    "tutor",
            "escalate": "escalate",
        }
    )

    # Persist state to SQLite — every turn is checkpointed
    memory = SqliteSaver.from_conn_string("memory.db")



    memory = InMemorySaver()

    return builder.compile(checkpointer=memory)


# ── Public interface ────────────────────────────────────────────

graph = build_graph()


def chat(
    session_id: str,
    student_name: str,
    subject: str,
    message: str,
    is_quiz_answer: bool = False,
) -> dict:
    """
    Send a message to the tutor graph and get a response.
    session_id ties all turns together — same student = same session_id.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Load existing state for this session (or start fresh)
    existing = graph.get_state(config)
    current_state = existing.values if existing.values else {}

    # Determine which node to start from
    # If the last turn left a quiz pending, student is answering it
    if is_quiz_answer and current_state.get("quiz_pending"):
        starting_node = "evaluate"
    else:
        starting_node = "tutor"

    input_state = {
        "messages":      [HumanMessage(content=message)],
        "subject":       subject or current_state.get("subject", "General"),
        "topic":         current_state.get("topic", ""),
        "confidence":    current_state.get("confidence", 5),
        "quiz_pending":  current_state.get("quiz_pending", False),
        "escalated":     False,
        "student_name":  student_name or current_state.get("student_name", "Student"),
    }

    result = graph.invoke(input_state, config=config)

    # Extract the last AI message as the reply
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_messages[-1].content if ai_messages else "Sorry, something went wrong."

    return {
        "reply":        reply,
        "topic":        result.get("topic", ""),
        "confidence":   result.get("confidence", 5),
        "quiz_pending": result.get("quiz_pending", False),
        "escalated":    result.get("escalated", False),
    }