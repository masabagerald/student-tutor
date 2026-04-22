import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from graph import chat

load_dotenv()

app = FastAPI(title="Student Tutor")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):  

    return templates.TemplateResponse(
        request,
        "index.html"
)


@app.post("/chat")
async def chat_endpoint(
    session_id: str = Form(...),
    student_name: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
    is_quiz_answer: bool = Form(default=False),
):
    try:
        result = chat(
            session_id=session_id,
            student_name=student_name,
            subject=subject,
            message=message,
            is_quiz_answer=is_quiz_answer,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/new-session")
async def new_session():
    """Generate a fresh session ID for a new student."""
    return JSONResponse({"session_id": str(uuid.uuid4())})