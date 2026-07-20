# What is Claude Code?

Claude Code is an agentic coding tool that understands your code base, edits your files, runs commands and integrates with your existing development tools and helps you get things done faster. It's available in your terminal, a desktop app, on the web, in VS Code and JetBrain IDEs.

Unlike Claude AI (which is a chat application similar to OpenAI or Gemini Chat), Claude Code has direct access to all your files in your terminal and your entire codebase. So, instead of copying & pasting code back & forth, it can go in and do all the work for you. The easist differentiator is that **Claude Code works as an AI Agent**

## What is an AI Agent?

An AI agent is a **software that can interact with its environment and perform actions to complete a defined goal**. The most basic way this can be done is by having a LLM run in a loop in real-time. **AI agents have access to tools, external services and other agents to help it reach its predefined goals**.

<div align="center">
    <image src="images/claude_code.png" alt="Claude Code"/>
</div>

## What can Claude Code actually do?
* It can read and understand your codebase.
* You can ask it to explain a feature or some dense code.
* You can ask it to trace & fix a bug throughout your code
* You can ask it to explain some unexpected output - it will analyze your codebase & tell you exactly why what you see is happening.
* It can execute your build script, run your tests, install packages and use the output to decide what to do next.
* It can search the web, if it needs access to the latest API documentation for example.

## Using Claude Code Effectively

To use Claude Code effectively, it's important to know the following concepts:

1. The **context window** (`/context`): think of this as Claude's working memory. It can hold a lot, but not everything at once. This is where the agentic aspect of it comes in. Finding strategic ways to find the answers within your codebase without stroring your entire codebase into context.
2. It **asks for permission**. By default, Claude Code will ask you for confirmation/permission before running commands, or making changes to your codebase. You're _always in control_, whether that's being more hands-on (I'll review everything before it happens!) or passive (I'll let AI make the call!)
3. It **can make mistakes**! Just like any tool, Claude Code isn't perfect. It might misunderstand your intent, introduce a new bug or over-engineer a solution.

## How Claude Code Works

We know that Claude Code is different from usual chat applications. But, how does it work?

### The Agentic Loop

Claude Code is best explained through and Agentic Loop.

<div align="center">
    <image src="images/agentic_loop.png" alt="Agentic Loop"/>
</div>

1. _You enter a prompt_ into Claude Code.
2. _Claude Code will then gather context_ to complete your prompt - by interacting with the model, which will return text, OR with tool call(s) that Claude Code can execute.
3. _Then it takes action_, for example editing a file or running a command.
4. _Finally, it verified those results and determines if they achieved what your prompt set out to do_ in the first place.
5. If they do, then Claude finishes and waits for the next prompt from the user. 
6. If they don't, Claude goes back and runs the loop (2, 3, 4) again (and again) until the results are complete and verifiable.
7. Throughout this interaction, you are able to _add context_ or _interrupt it_ or steer the model to guide it towards your end goal.

### Context

Claude Code has a context window (`/context`), which determines how much of your conversation, file contentx, command outputs and more it can store and look back on. Once yoi reach that limit, Claude code compacts your conversation, which automatically determines what it can take out of the context window and what it can summarize in order to bring the context window down.


<div align="center">
    <image src="images/claude_code_context.png" alt="Claude Code Context"/>
</div>

### Tools

Tools are the backbone of how agents works. Currently most AI agents are simply input text and output text. Tools let Claude code and other agents determine when to execute code to get closer to a task. This could be a read-file tool, or search-web tool for example. Claude code uses semantic searching to determine when to call a tool and get the output of the tool.

### Permissions

Claude code also has permission modes. 

* **Default** mode is that _it needs to ask explicit permission before editing a file or running a shell command_ (every time!). Use can use `Shift + Tab` to toggle between various permission modes on Claude Console. 
* **Auto Accept** edits files without asking, but still asks before executing command.
* **Plan mode** uses read-only tools to help compile a plan of action before starting.

It's worth being cautious when setting permissions. Giving Claude Code free reign to execute commands means a mistake could be harder to catch.

Claude Code works by combining various Agentic AI concepts:

* An Agentic Loop
* A managed context window
* Tools, and 
* Configurable permissions into your terminal

It can read your codebase, take actions and verify its own work - and that makes it fundamentally different from a chat window (like Chat GPT)

## Installing Claude Code

Claude code is simple to install, whether you want to use it in a terminal, the web or your IDE.

**NOTE:** in all the example commands below, I have used `$>` to denote your command prompt. Don't type the `$>`. Type in only whatever follows the `$>`.

### Terminal

**Mac OS/Linux or WSL**

Use the following command from the shell to install Claude code in one go:

```bash
$> curl -fsSL https://claude.ai/install.sh | bash
```

On MacOS, you can also use HomeBrew to install Claude Code, but note that it does not have auto-update capability.

**Windows**

In **PowerShell**, run the following command
```powershell
$> irm https://claude.ai/install.ps1 | iex
```

Alternately, if you are on **cmd command terminal**, you can also use the following command

```cmd
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

or using `winget`, type in the following: 

```cmd
$> winget install -e --id Anthropic.ClaudeCode
```

Just like Homebrew, it will not auto-update.

### Running Claude Code

* After installing Claude code for your respective OS, start a terminal/command prompt window and navigate to your code folder.
* Type in `claude` at the command prompt to start Claude Code in your project directory. The very first time you run this, it will ask you for some preferences (such as mode - light/dark etc.; color-theme to use etc.)
* Then sign in with your claude account (Pro/Max or Entrprise) OR you can use an API key from Anthropic. 
* Whichever directory you just ran Claude Code in, it will have access to all files & folders in that directory and all its sub-directories.

### VS Code & JetBrians IDEs

* In VS Code, open up the extensions panel and search for `Claude Code for VS Code`. Make sure it's from Anthropic, and install an extension as usual.
* Once installed, you should see the Claude Code icon in the top-righ portion of the active tab of your editor. 
* Click on the tab and follow the instructions.
* Similarly, for JetBrains IDEs (like Pycharm), install the Claude plugin from the JetBrains markeplace and follow instructions to configure it. (**NOTE:** it could be a BETA version, but check that it's from Anthropic before installing)
* In VS Code, Claude Code can run in Terminal experience or side panel. In JetBrains IDEs it runs only in Terminal experience. Ironically, the Terminal experience is most popular with developers.

## Your First Prompt

You talk to Claude Code just as you would talk to any AI Assistant (or chat application, such as ChatGPT). For example, here is a prompt to create a complete application:

```bash
Create a FastAPI project that lets me talk with the claude API
```

Here are some things to consider that will protect and make things easier for you:

* **Auto Accept Approval** you can choose whether Claude Code auto accepts every file change it suggests or require it to ask you for explicit permission each time. Use `Shift+TAB` to cycle between both modes on Terminal Experience.
* **Plan Mode** within the `Shift+TAB` menu (keep pressing `Shift+TAB` to toggle between options) is the Plan Mode, which takes your command and uses read-only tools to analyze your code-base, and do research on your suggested implementation. It will also ask you questions on items it wants clarifications on. It then _returns to you a long detailed plan, that it can execute_ on the codebase, when you ask it to. Plan mode is great to plan out complex code changes on a codebase or doing a safe code review.

When using Claude Code try to be as descriptive as possible with your prompt.

## Daily Workflows

If there is one thing you want to take from Claude Code, let it be this workflow:

```
Explore -> Plan -> Code -> Commit
```

Without this, most people ask Claude to directly generate code, which usually means a lot of course correction along the way.

### Explore and Plan

The fastest way to handle Step 1 & 2 in the workflow above is with **Plan Mode**. In Plan Mode, Claude Code cannot edit files. It just reads files to research about how to tackle the implementation of user's request. [Hit `Shift+Tab` key in Terminal experience, until you see plan mode]

For example, here is a request you could make in Plan model:

```
> I need to add webp conversion to our image upload pipeline. Figure out where in the pipeline it should happen, whether we need new dependencies and how to approach it.
```

And Claude Code will go off, read your entire code base, refer to the web and return with a complete plan of action - no changes to code files! At this point, you can review the recommendations and determine if it meets your criteria and accept/or ask Claude to do something else. For example:

```
> Can you limit the uploads to 10Mb?
```

You can also use the `Explore` command to ask Claude Code to explore your codebase _without_ being in the _plan mode_.

### Code

Now once the plan looks good, you can select "Approve" to accept the plan and let Claude handle all the items in the Plan it provided. Claude will review the codebase (new code it generates or code it updates) before it considers the plan as finished.

## Customizing Claude Code

### The CLAUDE.md file

One of the most useful parts of Claude Code is the `CLAUDE.md` file (name **is** case-sensitive!). It gives Claude Code persistent memory about your Claude project. 

When you open up Claude Code without a `CLAUDE.md` file, it's like it has to start afresh every single time. It has to re-explore your codebase, understand the dependencies are needed and the features that are already implemented. Sometimes it has to make assumptions, which makes it harder for us to steer Claude in the right direction. That's where `CLAUDE.md` file comes in.

It's a markdown file that you add to the root of your project and Claude Code reads it automatically everytime you start a session in the project's folder. It's like an onboarding script for your codebase. **Simply put, the contents of CLAUDE.md file are appended to your own prompt each time**

You can run the `/init` command which will make Claude generate one off your codebase.

Here is a sample CLAUDE.md file for a web application:

```markdown
# Project
This is a Next.js 15 app using the App Router, Tailwind and Drizzle ORM

# Commands
- Dev Server: `pnpm dev`
- Run tests: `pnpm test`
- Lint: `pnpm lint`

# Code Style
- Use 2 space indentation
- Prefer named exports
- All API routes go in app/api
- Use server actions instead of API routes where possible
```

Now when I ask Claude Code to create a React component, it knows how to style it with Tailwind or any other CSS framework that I'll be using. We see that Claude Code does a better job at doing its job right off the bat. First is having to understand where everything is at first.

#### Sharing your CLAUDE.md file

You can (it is recommended) share this file in your version control system for your team to use. But there is a hierarchy of memory files depending on who it is for.

* **Project User:** `project/CLAUDE.md` - this is a project level file that _lives in the root directory of your project_. <br/>It **serves as the standing system prompt and persistent memory layer for Claude Code**. It automatically injects project guidelines, build commands, and tech stack contexts straight into Claude's memory at the start of every session so you do not have to repeat context every time you prompt.<br/>If you have any files in your project folder that you want Claude to refer to, add them to this file with a `@` prompt. For example:

```
Database connection info is in @docs/database.md
Architecture definition is in @docs/architecture.md
```

* **User Level:** `~/.claude/CLAUDE.md (Mac or linux) or %USERPROFILE%\.claude\CLAUDE.md (Windows)` - this is a user-level CLAUDE.md file that lives in the root directory of your configuration folder.<br/> This file **holds your personal, cross-project preferences**. If you want Claude to always use a sarcastic humor style, write tests before implementation across every app you build, or avoid code comments, you define it here.


It is recommended that you start a new project WITHOUT a CLAUDE.md file, so you can see where you have to constantly course correct the Claude model. This keeps your CLAUDE.md file compact and contain only the necessary information that Claude can work with. The difference between a frustrating Claude Code session and a productive one comes down to the context. And CLAUDE.md is how you provide that context.

Start with your tech-stack, your preferenced and your commands [to build, test etc.] and build from there as you go.

### Sub-Agents

Sub-agents are specialized assistants that Claude Code can delegate tasks to. Each sub-agent runs within it's own conversation context window, with a custom system prompt that you define. When finished, it returns a summary to the main thread while all the intermediate work stays isolated. One  of the main advantages of sub-agents is that they help manage context window usage.

#### The main context window and sub-agent context window

When you chat with Claude Code, you are adding context to the main context window. Every tool call and its results get saved in this main context window. And so, when Claude uses a sub-agent, a separate windows starts. The sub-agent receives 2 inputs - a custom system prompt from your configuration file and a task description writtem by the parent or parent-agent based on what you asked for. The sub-agent then works autonomously when it reads files, edits files, uses tools - none of these will appear in the main conversation. Just a summary is returned. The entire sub-agent conversation then gets discarded.

#### What this means in practice

Consider a task of investigating how the payments system works in an unfamiliar codebase - maybe your are trying to use Claude Code to investigate which service handles refunds. Without a sub-agent, Claude might investigate 15 files, run several searches and trace through several function calls. All of that context fills your context window, even if you only needed 1 single fact - which service handles refunds. 

With a sub-agent, you get the answer, without the journey. The sub-agent explores, discovers the answer, and returns a focused summary keeping your main context clean! The main agent loses visibility on how the sub-agent reached the conclusion and what it discovered along the way.

#### Built-in sub-agents

Claude Code includes several built-in sub-agents that you can use immediately, like:

* **General-purpose sub-agent**: used for multi-step tasks that require both exploration and action.
* **Explorer sub-agent**: which is used to explore codebases.
* **Plan sub-agent**: used during plan mode for research and analysis of your codebases before presenting a plan.
* **Custom agents**: And you can also create your own sub-agent with custom system prompt and tool access.

Sub-agents let Claude Code break work into focused pieces, keep your main context window clean and bring back just what you need. Whether you're using built-in ones or creating your own, they are a practical way to get more out of longer Claude Code sessions.

### Skills

Every time you explain your team's coding standards to Claude, you are repeating yourself. Every PR review, you redscribe how you want feedback structured. Every commit message, you remind Claude of your preferred format -- and **Skills fix this**.

A skill is a markdown file that teaches Claude how to do something once, and Claude applied that knowledge automatically whenever it's relevant. 

Agent skills are folders of instructions, scripts and resources that agents can discover and use to do things more accurately and efficiently. Thes sub-folders are created under the `.claude` sub-folder under your project's root folder (see image below). The sub-folder name is your skill name (e.g. `pr-review`), and this name is case-sensitive.

<div align="center">
    <image src="images/claude_code_skills.png" alt="Claude Code Skills"/>
</div>

Within each skill folder (such as `pr-review`) we have a `SKILL.md` (name is case sensitive!) markdown file. At the top of each `SKILL.md` file you must have entries for `name` and `description` as shown in example below:

```markdown
---
name: pr-review
description: Reviews pull requests for code quality. Use when reviewing PRs or checking code changes
---
```

The `description` field is how Claude Code decides whether to use the skill. When you ask Claude to "review this PR", it matches your request against available skill descriptions and finds this one. Claude reads your request, compares it to all available skill and activates the ones that match.

#### Where Skills live

You can store skills in a few places depending on who needs them. 

* **Personal Skills** go in the `~/.claude/skills` folder. As described above, each skill gets its own sub-folder under the `~/.claude/skills` folder - for example, the `commit-message` _personal skill_ will be described in `~/.claude/skills/commit-message/SKILL.md` file. Personal skills follow you across all your projects. These are usually your preferences, your commit messages, your documentation formats, how you like code explained.
* **Project Skills** go under the `.claude/skills` subfolder created off your project's root folder. As with personal skills, each project skill gets its own sub-folder. For example our `pr-review` project skill will be written in the `.claude/skills/pr-review/SKILL.md` file. These skills will be checked into version control and will be available to anyone that clones your repo. This is where team standards live, like your company's brand guidelines, preferred fonts & colors you use for web design for example.

Each skill (Personal or Project) can be called _explicitly_ using the slash command that includes the skills's name - for example `/commit-message` or `/pr-review`.

#### What makes Skills different?

Claude code has several ways to customize behavior. Skills are unique because they are automatic and task specific. `CLAUDE.md` files loads into every conversation - for example, if you _always_ want Claude Code to use TypeScript scrict mode, that instruction will go into your `CLAUDE.md` file. Skills on the other hand load on-demand, when they match your request. It _only loads in the name & description_, so it does not fill up your entire context window. For example, your entire PR checklist instructions need not be in your context when you are debugging - it will load only when you actually ask for it. Unlike calling a skill explicitly using a slash (`/`) command, Claude will automatically apply the respective skill when it recognizes the situation.

#### When to use Skills?

Skills work best for specialized knowledge that applies to specific tasks. For example, code review skills your team follows; commit message formats that you prefer, brand guidelines of your organization. If you find yourself explaining the same thing to Claude repeatedly, that's a skill waiting to be written!

### Model Context Protocol (MCP)

MCP is a protocol is an open standard that helps Claude Code connect to external data sources and tools. When you ask a question, Claude Code will automatically understand when it should use those tools to better understand your query. Context is one of the most important parts when working with Claude Code. A lot of your context lives elsewhere, like in your databases, your productivity apps, or in public repositories. This is where MCP comes in.

#### What can you do with MCP?

First, it's important to understand the concept of tools when talking about Agentic AI. Tools give agents like Claude Code the ability to perform actions in order for them to better complete their tasks. This is different from other AI where you just get an output directly in text usually. For example, if your team is using `Linear` as your project management software you can add in a `Linear` MCP server to bring in the details of your specific issues. If you want to get up-to-date documentation of the dependency you are working on, then the `context7` MCP server will provide Claude Code with that. 

Refer to 100's of available MCP servers at [claude.com/connectors](https://claude.com/connectors).

#### Adding MCP servers

You can add a MCP server with the Claude `claude mcp add` command. There are 2 main types:

* **HTTP Servers** are for remote services, which are hosted by the service provider and connect over the network. 
* **STDIO Servers** are for local processes that run on your machine. 

You can manage your MCP servers with a `/mcp` command inside your Claude Code session to see what's connected, the status and to disable servers that you don't want to use. 

#### Scoping MCP Servers

MCP servers can be scoped in three different ways:

* **Local**: it's only available in the current project for you. 
* **User**: available across all _your_ projects (defined in `~/.claude.json`)
* **Project**: uses a `project_root/.mcp.json` that you check into your version control so anyone working on the codebase gets the exact same servers automatically. 

#### Context Costs

One thing to be aware of is that MCP servers add tool definitions to your context window, _even when you are not using them!_So if you have a lot of MCP servers configured, it will eat into your context. Run the `/mcp` command to check which servers are connected and disable anything that you are not actively using. 

If a tool has a CLI equivalent (like GH for Github), it is more context efficient because it does not add persistent tool definitions. You could also benefit by using a skill in this scenario. The skill's name and description will be loaded into the context, similar to MCP. When Claude thinks it needs to use that skill, it will then load it into the context window, which is where you could put in the command line interface tool. 

If your MCP tools exceed 10% of your context window, Claude Code will automatically switch to tool-search mode, which will discover the right tools on demand, but this might not work as well just because it's not in the context. 

### Hooks

Hooks let you run commands at different times in Claude Code's lifecycle. The key difference between hooks and everything else we learnt so far is that hooks are deterministic - they will always run!

#### Why use hooks?

You could tell Claude Code, in your `CLAUDE.md` file to run `prettier` after every file edit or run `eslint` after every tool call; and most of the time it will work - but it may not run always, because it's not perfect. But a hook makes it happen every single time with no exceptions. Common use-cases could include running a formatter after file edits, logging all executed commands for compliance, blocking dangerous operations like modifying production files, and sending yourself notifications when Claude finishes as task.

#### How Hooks work

Hooks are configured in your `settings.json` file (in IDEs like VS Code or its forks, such as AntiGravity or Cursor). You pick an event (such as `PreToolUse`), optionally set a matcher for which tool it applies to, and provide a command to run (as shown in the snippet below). You configure them through the `/hooks` command inside Claude Code.

```json
"hooks" : {
    "PreToolUse": [
        "matcher" : "Bash",
        "hooks" : [
            {
                "type" : "command",
                "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/block-dangerous-commands.sh",
            }
        ]
    ],
    "PostToolUse" : [
        // fire whenever Claude modified a file
        "matcher" : "Edit|MultiEdit|Write",
        "hooks" : [
            {
                // run appropriate formatter
                "type" : "command",
                // the shell script could run any applicable
                // formatter for the edited file.
                "command" : "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/auto-format.sh",
                "timeout" : 20,
            }
        ]
    ],
```

| Tool | When Does it Run? |
| --- | --- |
| `UserPromptSubmit` | Runs when you submit a prompt, but _before_ Claude processes it. |
| `PreToolUse` | Runs before a tool call. |
| `PostToolUse` | Runs after a tool call completes. |
| `Notificaton` | Runs when Claude sends a notification. |
| `Stop` | Runs when Claude finishes responding. |

**Let's look at a practical example**:

The most common eample is code formatting after edits - see an example of `PostToolUse` in the JSON file above, which sets up a command to format file (or multiple files) post edit. The shell script `auto-format.sh` file could run an appropriate formatter depending on the file being edited - for example, `prettier` for TypeScript, `ruff` for Python as so on.

#### Blocking with PreToolUse

`PreToolUse` hooks can block tool calls before they execute. Your hook receives the tool name and input as JSON on `stdin`. The exit code determines the behavior:

* **Exit code 0** — proceed normally.
* **Exit code 2** — _block the action_. The `stderr` message gets fed back to Claude as feedback so it knows why it was blocked and can adjust.
* **Any other exit code** — a non-blocking error that gets shown to you but doesn't stop anything.

This is how you enforce hard rules. Block writes to a production config directory. Block bash commands that contain `rm -rf`. Block commits to `main`. Whatever your team needs to be guaranteed, not suggested.

Following shows an example in `settings.json`:

```json
{
    "hooks" : {
        "PreToolUse" : [
            {
                "matcher" : "Bash",
                "hooks" : [
                    "type" : "command",
                    "command" : "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/block_dangerous_commands.sh",
                ]
            }
        ]
    }
}
```

#### Sharing Hooks with Your Team
Hooks configured in `.claude/settings.json` are project-level and can be checked into your repo. This means your entire team gets the same hooks automatically. Use the `CLAUDE_PROJECT_DIR` environment variable in your commands to reference scripts stored in your project, so they work regardless of Claude's current working directory.

#### Recap
Hooks give you deterministic control over Claude Code's behavior. Use `PostToolUse` for auto-formatting and logging. Use `PreToolUse` to block dangerous operations. Configure them with `/hooks` or in `settings.json`. And check them into your repo so your team gets them too.

If something needs to happen every time without fail, don't put it in a prompt. Put it in a hook.