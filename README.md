# Learning LangChain
Learning LangChain from a variety of sources

## Setting up your virtual environment
I have used `uv` to manage the Python environment for this project. Here are the steps you should follow to recreate the same Python environment at your end, after you pull this GitHub repo to your local folder.

Before re-creating the Python environment, ensure you have `uv` installed on your local machine. To install `uv` for your operating system, follow the instructions [here](https://docs.astral.sh/uv/)

### Recreating the Python environment
These are the steps to setup the Python environment (**NOTE**: in the code listings below `$>` refers to the command prompt and should not be entered by you!)
1. `cd` to folder where you pulled the Github repo - let's refer to this folder as `$PROJECT_HOME` henceforth
    ```bash
    $> cd folder/to/learning_langchain
    ```
2. Run the following command inside `$PROJECT_HOME` - - this  creates a local Python environment in the `$PROJECT_HOME/.venv`, but _does not_ install any Python packages!
    ```bash
    $> uv venv --python 3.12
    ```
3. Activate the Python environment you just created
    ```bash
    $> source .venv/Scripts/activate  ## on a Mac/Linux

    $> .venv\Scripts\activate ## on Windows
    ```
4. Install the packages from `pyproject.toml` file
    ```bash
    $> uv pip install --editable .
    ```
5. Install some _global_ packages - these will be installed globally without _polluting_ your local Python environment. The include packages such as `ipykernel` (to run Notebook files) and `black` (to format Python code)
    ```bash
    $> uv tool install black 
    $> uv tool install ipykernel
    ```

### Running the sample programs
The sample programs are a mix of Notebook files (`*.ipynb`) and Python modules (`*.py`) files. 

To run the notebooks files (`*.ipynb`)
1. Open the notebook in your code editor, such as IPython Notebook or VS Code or Cursor or PyCharm (professional only!)
2. Select the correct Python kernel to run the cells in your notebook - it should be `.venv\Scripts\python.exe`
3. Run the notebook as you normally do.

To run Python modules (*.py) files:

From within your IDE -
1. Assign the correct Python interpreter to your project - it should be `.venv\Scripts\python.exe`
2. Run the Python code as you'd normally do from your IDE.

From the command line -
1. `cd` to `$PROJECT_HOME`
    ```bash
    $> cd folder/to/learning_langchain
    ```
2. Use `uv` to run your Python code as follows:
    ```bash
    $> uv run full/path/to/python/code/file
    ```
    For example:
    ```bash
    $> uv run src/langchain_tutorial/01_chat_models_and_prompts.py
    ```





