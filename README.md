# Learning LangChain
Learning LangChain from a variety of sources

## Setting up your virtual environment
I have used `uv` to manage the Python environment for this project. Here are the steps to setup the Python environment at your end assuming you have `uv` installed locally and you'll be downloading this Github repo to a local folder.

To install `uv` for your operating system, follow the instructions [here](https://docs.astral.sh/uv/)

These are the steps to setup the Python environment
1. `cd folder/to/learning_langchain` - let's refer to this folder as `$PROJECT_HOME`
2. `uv init .venv --python 3.12` - this will create a local Python environment in the `$PROJECT_HOME/.venv`, but _does not_ install any Python packages!
3. `uv pip install --editable .` - this will install all the packages listed in the `pyproject.toml` file and recreate the Python environment inside `$PROJECT_HOME/.venv` and it's subfolders.
4. Select the right Python interpreter 
* Inside your IDE: Open the folder in your development environment, for example VS Code or Cursor IDE. Use the Python interpreter  `.venv/Scripts/python.exe` folder and you are all set!
* On the command prompt: run the following commands on the command prompt:
```bash
$> cd $PROJECT_HOME
$> .venv\Scripts\activate (on Windows)
OR
$> source .venv/Scripts/activate (on Mac or Linux)
$> uv run <path to Python script file>
```





