# The Developer's Prompt Framework (CCSE)
CCSE is a four-part framework purpose-built for developer prompts. It stands for:

**Context**, **Constraints**, **Specification**, **Example**. Use it as a mental checklist for any
non-trivial request.

## C - Conte t: Set the Stage
**Tell Claude about your environment**: programming language, framework, existing code patterns, team conventions. Claude can't read your mind or your codebase. What you leave out, it fills in with reasonable defaults that may not match your reality.

Example:
```bash
# GOOD - Context rich
I have a Django 4.2 REST API with DRF. We use JWT auth via Django REST Framework & Simple JWT.

Our User model has email, is_staff, and organization_id fields.

# BAD: Context-empty
"I have a Dango API"
```