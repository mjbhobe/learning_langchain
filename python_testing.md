# Python Testing
**Software Testing** is the process of identifying bugs in an application by running it through a series of designed tests. 

It involves multiple stakeholders, such as testers, management, business, customers, consultants, and end-users.

**Software testing** is important for the following reasons:
* It can detect errors or defects made during development phase
* It increases reliability and quality of software.
* It validates or checks if software is behaving as expected based on requirements.
* It is required for effective performance of software applications.

The **Software Testing Life Cycle (STLC)** refers to various phases involved in testing of software. It generally includes:
* Requirements Review
* Test Planning
* Test Designing
* Test Environment Setup
* Test Execution:
    * Unit Testing
    * Integration Testing
    * System Testing
    * Regression Testing
    * Acceptance Testing
* Test Reporting

## Unit Testing
**Unit Testing** is testing the smallest possible code snippets of a program (usually functions or methods). Practically breaking down of these code snippets into further smaller pieces is not achievable. For example, consider the following function
```python
def square(n):
    """returns square of a number"""
    return n * n
```
## Integration Testing
**Integration Testing** focuses on testing interactions across units. These tests are still run in isolation in order to overcome inputs from outside or other units. Inputs required from outside are simulated.

## System Testing
**System Testing** checks parts of program once all units are plugged together. It is an _extreme form_ of integration testing. System tests will not be useful of these are not supported by results of integration tests and unit tests.

## Regression Testing
* **Regression Testing** is another type of testing which ensures that the previously built software performs the same way after making some changes to it or interfacing with other software.
* Regression tests can be written before or after bug is found. They provide an assurance that an increase in program's complexity does not introduce additional bugs.

## Acceptance Testing
* **Unit Testing**, **Integration Testing**, and **System Testing** are done to verify software product as a whole or it's individual components.
* On the other hand, the testing that you do to confirm that the software behaves as expected is known as **Acceptance Testing**
* **Acceptance Testing** DOES NOT test the core functionality of the software - i.e. does not test at a granular level. It only aims at checking the software behavior from the end-user perspective.

## Python Unit Testing Frameworks
* There are 4 major unit testing frameworks used with Python code: `doctest`, `unittest`, `nose` and `pytest`
* `doctest` is the simplest framework, which combines documentation & tests.
* `unittest` framework has xUnit-Style architecture.
* `nose` extends `unittest` and makes testing easier.
* `pytest` has a rich set of plugins and is compatible with both `unittest` and `nose`.

### DocTest
* `doctest` is a python testing module, which allows writing tests based on expected output from standard Python interpreter along with documentation.
* `doctest` is also known as `document testing` or `testable document`
* `doctest` allows combination of tests and documentation. This feature enables to keep documentation up to date with reality and ensures that tests express the expected behavior.

#### DocTest Specifications
* A `doctest` contains documentation and one or more valid Python statements
* The statements are written after Python shell's `primary primpt (>>>)`
* `Secondary prompt (...)` is used from second line onwards, when a statement is written across multiple lines.
* The `expected output` is written below the Python statements and should be the same as the result obtained, when you run the statement in a Python shell.

#### Example:
* Tests are written in the form of documentation in a text file. Let's see an example file `sample_tests.txt

```
================================
demonstrating usage of doctest
================================

1. This doctest checks functionality of Python's plus operator.

>>> 2 + 4
6
>>> -10.5 + 8
-2.5
```
* Presence of a blank line or a new line starting with primary prompt `(>>>)` after the expected output is seen as end of existing test.

#### Running the tests
* Run the tests using the following command
```bash
$> python -m doctest sample_tests.txt
```
* By default **no output is seen when all tests pass**. To view verbose output, add the `-v` option to above line.

#### Using Secondary Prompt
* Let's extend `sample_tests.txt` by appending function definition of `add2num` and two tests

```
2. The below doctest checks functionality of add2num function

>>> def add2num(x, y):
...     """this function returns sum of 2 numbers"""
...     return x + y
>>> add2num(6, 7)
13
>>> add2num(-8.5, 7)
-1.5
```
*This example shows how a function spanning multiple lines is defined using primary `(>>>)` and secondary `(...)` prompts in a text document.

### Writing tests in docstrings
* A more practical use of `doctest` comes from writing tests within `doc strings` in a Python file itself.
* We use the same syntax as explained above, just that they are written as part of the `doc string`
* Here is an example of writing tests in a `sample_module.py` Python file, where you can see 2 unit tests.
```python
def add2num(x, y):
    """ adds two given numbers an returns the sum
    >>> add2num(6,7)
    13
    >>> add2num(-8.5, 7)
    -1.5
    """
    return x + y
```

#### Running tests in docstrings
```bash
$> python -m doctest sample_module.py
```

#### Handling Exceptions
* In some cases you'd like to handle exceptions thrown from functions
* `doctest` can detect raised `exceptions` by detecting Python exception report and traceback displayed in interactive shell.
* `doctest` is concerned only with the first line `Traceback (most recent call last):` and the last line which tells it which exception you expect
* `doctest` will report failure only of one of these two parts doesn't match
* For example, we can handle exceptions like this
```python
def add2nums(a: int, b: int) -> int:
    """ adds 2 numbers and returns sum
    >>> add2nums(7, 20)
    27
    >>> add2nums(-17, 5)
    -12
    >>> add2nums(-17, "hello")
    Traceback (most recent call last):
    TypeError: unsupported operand type(s) for +: 'int' and 'str'
    """
    return a + b
```

### Handling explicit blank lines
* `doctest` treats the first blank line after the primary prompt `(>>>)` as the end of tests
* However, sometimes you expect blank lines in the output of your program and you want to tell `doctest` to expect blanklines in output.
* These can be indicated by inserting the word `<BLANKLINE>` in place of expected blank lines.
* Here is an example:
```python
def insertlines(n : int):
    """prints n blank lines between line printing Start and ending with End
    >>>insertlines(2)
    Start
    <BLANKLINE>
    <BLANKLINE>
    End
    """
    print('Start')
    for i in range(n):
        # print a blank line
        print()
    print('End')
```
