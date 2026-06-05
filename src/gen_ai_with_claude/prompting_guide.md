# Claude's Prompting Framework

## The Developer's Prompt Framework (CCSE)

CCSE is a four-part framework purpose-built for developer prompts. It stands for:

**Context**, **Constraints**, **Specification**, **Example**. Use it as a mental checklist for any
non-trivial request.

### C - Context: Set the Stage

**Tell Claude about your environment**: programming language, framework, existing code patterns, team conventions. Claude can't read your mind or your codebase. What you leave out, it fills in with reasonable defaults that may not match your reality.

Example:
```bash
# GOOD - Context rich
I have a Django 4.2 REST API with DRF. We use JWT auth via Django REST Framework & Simple JWT.

Our User model has email, is_staff, and organization_id fields.

# BAD: Context-empty
"I have a Dango API"
```

### C — Constraints: Set the Guardrails

What are the boundaries? Which libraries you can or can't use? Performance requirements?  Coding style rules? Lines-of-code limits? Response format requirements?

Example:

```bash
# GOOD: Clear constraints
Use only the standard libraries — no third-party packages.
Keep the function under 30 lines.
Type-annotate all parameters and return values
Write in a functional style — no classes.
```

### S — Specification: State the Goal Precisely

What exactly should the function/system do? Input and output types? Edge cases to
handle? Error behavior? The more specific you are, the less Claude has to guess.

### E — Eample: Sho Don't Just Tell

Whenever possible, provide an example of the pattern you want — an existing
function, a code style sample, or an input/output example. Examples are worth a
thousand words to Claude.

## Persona - the System Prompt

The system prompt is a special instruction that runs before every message in a
conversation. It's your opportunity to give Claude a permanent persona, coding
standards, and output format that persist across the whole session.

Example:n (notice how detailed this is)

```python
# Powerful system prompt for a Python engineering session
SYSTEM_PROMPT = """
You are a senior Python engineer with 10 years of experience.
You write clean, production-ready code that follows these standards:

CODING STANDARDS:
- PEP 8 style- Full type annotations (Python 3.10+ union syntax)
- Docstrings in Google format
- Comprehensive error handling with specific exception types
- Never use bare "except:" clauses

RESPONSE FORMAT:
- Lead with the code
- Follow with a brief explanation of key decisions
- Flag any assumptions you made
- Note any edge cases the implementation doesn't handle

TECH STACK CONTEXT:
 - Python 3.11, FastAPI 0.100+, SQLAlchemy 2.0, PostgreSQL 15
 - Redis for caching, Celery for background tasks
 - pytest for testing
"""
# Use in every API call:
client.ask(prompt=user_question, system=SYSTEM_PROMPT)
```