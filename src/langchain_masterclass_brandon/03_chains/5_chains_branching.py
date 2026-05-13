"""
chains_branching.py - example of chain branches depending on initial outcome

@Author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

# load all environment variables
load_dotenv()

# create my LLM - using Google Gemini
model = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# only for colorful text & markdown output support
console = Console()

# template to use if feedback is positive!
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thank-you note for this positive feedback: {feedback}"),
    ]
)

# template to use if feedback is negative!
negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}"),
    ]
)

# template to use if feedback is neutral!
neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}",
        ),
    ]
)

# template to use if feedback requures and escalation
escalation_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Generate a message to escalate this feedback to a human agant: {feedback}",
        ),
    ]
)

# template to use to classify the sentiment of the feedback/review
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}",
        ),
    ]
)

# the chain that classifies sentiment of the feedback
classification_branch = classification_template | model | StrOutputParser()

# branching off to various handlers depending on sentiment of feedback
routing_branches = RunnableBranch(
    (
        lambda x: "positive" in x.lower(),
        positive_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "negative" in x.lower(),
        negative_feedback_template | model | StrOutputParser(),
    ),
    (
        lambda x: "neutral" in x.lower(),
        neutral_feedback_template | model | StrOutputParser(),
    ),
    # for all else, escalate
    escalation_feedback_template | model | StrOutputParser(),
)

# chain that ties them together
chain = classification_branch | routing_branches

# here are sample feedbacks
positive_feedback = """
I had a great experience with CleenoBot. It did a fantastic job of cleaning all floors, especially around the corners and in hard to reach places. I would highly recommend it to others. Thank you for such a wonderful product!"""

negative_feedback = """
I has high expectations of CleenoBot, especially after seeing the demo. However I am dissappointed with the quality of cleaning. It hardly picks up any dust or dirt off the carpet and can hardly reach the corners of my room. I am returning this & will not recommend this to anyone."""

neutral_feedback = """
I have been using the CleenoBot for a week now. It performs the basic cleaning tasks as described in the manual and handles most floor surfaces adequately. The battery life is consistent with the product specifications provided. It is a functional appliance that performs its intended job without 
any major issues or standout features."""

escalation_feedback = """
This CleenoBot is an absolute safety hazard and a total scam. During its first run, the battery overheated 
to a dangerous level and left visible scorch marks on my hardwood floors—it’s a miracle it didn’t start a fire. Your customer service line is a joke and no one is taking responsibility for the property damage. 
I am demanding an immediate full refund and want to speak with a senior manager today. If this isn't resolved within 24 hours, I will be filing a formal complaint with the Consumer Product Safety Commission and contacting my legal counsel. Absolutely pathetic product and even worse company."""

# console.print(f"[green]Positive Review:[/green] {positive_feedback}\n")
# response = chain.invoke({"feedback": positive_feedback})
# console.print(Markdown(response))

# console.print(f"[yellow]Negative Review:[/yellow] {negative_feedback}\n")
# response = chain.invoke({"feedback": negative_feedback})
# console.print(Markdown(response))

# console.print(f"[blue]Neutral Review:[/blue] {neutral_feedback}\n")
# response = chain.invoke({"feedback": neutral_feedback})
# console.print(Markdown(response))

# this piece of code shows you how the classification chain works
# for each type of feedback

# for feedback in [
#     positive_feedback,
#     negative_feedback,
#     neutral_feedback,
#     escalation_feedback,
# ]:
#     response = classification_branch.invoke({"feedback": feedback})
#     console.print(f"[blue]Feedback: [/blue] {feedback}")
#     console.print(f"[cyan]Classification: [/cyan] {response}")

# now let's see how thw chain works for each type of feedback
for feedback in [
    positive_feedback,
    negative_feedback,
    neutral_feedback,
    escalation_feedback,
]:
    response = chain.invoke({"feedback": feedback})
    console.print(f"[blue]Feedback: [/blue] {feedback}")
    console.print(f"[cyan]Email generated: [/cyan] {response}")
    print("--" * 50 + "\n\n")


# console.print(f"[red]Escalation Review:[/red] {escalation_feedback}\n")
# response = chain.invoke({"feedback": escalation_feedback})
# console.print(Markdown(response))
