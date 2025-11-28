# Python Testing

**Software Testing** is the process of identifying bugs in an application by running it through a series of designed
tests.

It involves multiple stakeholders, such as testers, management, business, customers, consultants, and end-users.

**Software testing** is important for the following reasons:

* It can detect errors or defects made during development phase
* It increases reliability and quality of software.
* It validates or checks if software is behaving as expected based on requirements.
* It is required for effective performance of software applications.

The **Software Testing Life Cycle (STLC)** refers to various phases involved in testing of software. It generally
includes:

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

**Unit Testing** is testing the smallest possible code snippets of a program (usually functions or methods). Practically
breaking down of these code snippets into further smaller pieces is not achievable. For example, consider the following
function

```python
def square(n):
    """returns square of a number"""
    return n * n
```

## Integration Testing

**Integration Testing** focuses on testing interactions across units. These tests are still run in isolation in order to
overcome inputs from outside or other units. Inputs required from outside are simulated.

## System Testing

**System Testing** checks parts of program once all units are plugged together. It is an _extreme form_ of integration
testing. System tests will not be useful of these are not supported by results of integration tests and unit tests.

## Regression Testing

* **Regression Testing** is another type of testing which ensures that the previously built software performs the same
  way after making some changes to it or interfacing with other software.
* Regression tests can be written before or after bug is found. They provide an assurance that an increase in program's
  complexity does not introduce additional bugs.

## Acceptance Testing

* **Unit Testing**, **Integration Testing**, and **System Testing** are done to verify software product as a whole or
  it's individual components.
* On the other hand, the testing that you do to confirm that the software behaves as expected is known as **Acceptance
  Testing**
* **Acceptance Testing** DOES NOT test the core functionality of the software - i.e. does not test at a granular level.
  It only aims at checking the software behavior from the end-user perspective.

## Python Unit Testing Frameworks

* There are 4 major unit testing frameworks used with Python code: `doctest`, `unittest`, `nose` and `pytest`
* `doctest` is the simplest framework, which combines documentation & tests.
* `unittest` framework has xUnit-Style architecture.
* `nose` extends `unittest` and makes testing easier.
* `pytest` has a rich set of plugins and is compatible with both `unittest` and `nose`.

### DocTest

* `doctest` is a python testing module, which allows writing tests based on expected output from standard Python
  interpreter along with documentation.
* `doctest` is also known as `document testing` or `testable document`
* `doctest` allows combination of tests and documentation. This feature enables to keep documentation up to date with
  reality and ensures that tests express the expected behavior.

#### DocTest Specifications

* A `doctest` contains documentation and one or more valid Python statements
* The statements are written after Python shell's `primary primpt (>>>)`
* `Secondary prompt (...)` is used from second line onwards, when a statement is written across multiple lines.
* The `expected output` is written below the Python statements and should be the same as the result obtained, when you
  run the statement in a Python shell.

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

* Presence of a blank line or a new line starting with primary prompt `(>>>)` after the expected output is seen as end
  of existing test.

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

*This example shows how a function spanning multiple lines is defined using primary `(>>>)` and secondary `(...)`
prompts in a text document.

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
* `doctest` can detect raised `exceptions` by detecting Python exception report and traceback displayed in interactive
  shell.
* `doctest` is concerned only with the first line `Traceback (most recent call last):` and the last line which tells it
  which exception you expect
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
* However, sometimes you expect blank lines in the output of your program and you want to tell `doctest` to expect
  blanklines in output.
* These can be indicated by inserting the word `<BLANKLINE>` in place of expected blank lines.
* Here is an example:

```python
def insertlines(n: int):
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

### Using `unittest`

* `unittest` is another Python _built-in_ testing module (i.e., you don't have to install any additional modules to use
  it!)
* `unittest` is Python xUnit-like testing framework - it's functionality is similar to `JUnit` (for Java), `CPPUnit` (
  for C++) etc.
* All such frameworks are derived from SmallTalk's `SUnit` framework.

#### Architecture of a xUnit-style library

A xUnit style library contains the following components:

* `Test case class`: the base class of all test classes written in test modules
* `Test fixtures`: These are methods run before and after execution of test code
* `Assertions` are functions or methods that check the behavior of variables being tested.
* `Test Suite` is the collection of related tests that can be executed together.
* `Test Runner` is the block of code that runs tests
* `Test result formatter` formats test results output in human readable formats, such as plain text, HTML, XML etc.

#### Using `unittest`

Following code shows a typical implementation of test cases using `unittest`. Note that all of these are written in a
Python module

```python
# test_add2nums.py - unit tests for add2nums function

# assume our add2nums() function is written 
# in a module sample_module.py
from sample_module import add2nums

import unittest  # required!

"""
NOTE:
1. You MUST derive this class from `unittest.TestCase` base class
2. Each testcase is a method of this class - the method MUST start with `test_` & rest of the signature can be of your choice
3. Modify the main execution block to call `unittest.main()`
4. Run this module as `python -m unittest test_add2nums.py [-v]` (the -v is for verbose output)
"""


class Testadd2num(unittest.TestCase):
    # Happy path - positive tests
    def test_sum_2pos_nums(self):
        self.assertEqual(add2num(6, 8), 14)

    def test_sum_pos_and_neg_nums(self):
        self.assertEqual(add2num(6, -8), -2)

    def test_sum_num_and_zero(self):
        self.assertEqual(add2num(6, 0), 0)

    # here is how to tackle known exceptions thrown
    # from function being tested 
    def test_known_exception(self):
        """Test that TypeError is raised when passing a string."""
        # The built-in '+' operator for int and str raises TypeError
        with self.assertRaises(TypeError) as e
            # will raise TypeError 
            # - suppose our message is "unsupported operand type(s) for +: 'int' and 'str'"
            add2nums(17, "hello")
        self.assertEqual(str(e.exception), "unsupported operand type(s) for +: 'int' and 'str'")

        def test_typeerror_on_list_input(self):
          """Test that TypeError is raised when passing a list."""
          # assume our message is "unsupported operand type(s) for +: 'list' and 'int'"
          with self.assertRaises(TypeError) as e:
              add2nums([1, 2], 5)
          self.assertEqual(str(e.exception), "unsupported operand type(s) for +: 'list' and 'int'")


# finally to run this module modify "main()" as follows
if __name__ == "__main__":
    unittest.main()
```

#### Test Fixtures

* `Test Fixtures` are special methods called before and after the tests and these are overloaded methods of the
  `unittest.TestCase` class.
* Often some startup code - such as connecting to a database - needs to run before a test-module (the *.py file) OR the
  unit-test class (class derived from `unittest.TestCase`) OR the test-case itself is run.
* For each startup method, there is a corresponding _tear down_ method, which is used to clean up after the test (e.g.
  close database connection)
* The major test fixtures are shown below (**NOTE**: names are case-sensitive!)

##### Module level fixtures

| Fixture            | 	When is it called?                                                        | 	How to Define?                                                                     |
|:-------------------|:---------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| `setUpModule()`    | 	Before any test class in the module is loaded or any tests are run.       | Define as a standalone function directly in the module file (e.g., test_my_app.py). |
| `tearDownModule()` | 	After all test classes and all tests in the module have finished running. | 	Define as a standalone function directly in the module file.                       |

```python
# test_module.py - testing module
def setUpModule():
    # Setup for the whole module
    print("\n--- MODULE SETUP: Connecting to a shared external resource. ---")
    global EXTERNAL_RESOURCE
    EXTERNAL_RESOURCE = "Database Connection Handle"


def tearDownModule():
    # Cleanup for the whole module
    print("\n--- MODULE TEARDOWN: Disconnecting the shared external resource. ---")
    global EXTERNAL_RESOURCE
    del EXTERNAL_RESOURCE


class MyTests(unittest.TestCase):
    # ... test methods use EXTERNAL_RESOURCE ...
    pass
```

##### Class level fixtures

| Fixture           | 	When is it called?                                                                             | 	How to Define?                                         |
|:------------------|:------------------------------------------------------------------------------------------------|:--------------------------------------------------------|
| `setUpClass()`    | 	**Before** any test methods within that specific unittest.TestCase class are run.              | Define as a class method decorated with `@classmethod`. |
| `tearDownClass()` | 	**After** all test methods within that specific unittest.TestCase class have finished running. | Define as a class method decorated with `@classmethod`. |

```python
class TestMathFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup for the whole class
        print("--- CLASS SETUP: Creating a test user object. ---")
        # Store the resource on the class itself using 'cls'
        cls.test_user = {"id": 1, "name": "TestUser"}

    @classmethod
    def tearDownClass(cls):
        # Cleanup for the whole class
        print("--- CLASS TEARDOWN: Deleting the test user object. ---")
        del cls.test_user

    def test_user_exists(self):
        # Access the resource using 'self.test_user' or 'self.__class__.test_user'
        self.assertIsNotNone(self.test_user)
```

#### Method level fixtures

| Fixture      | 	When is it called?                                           | 	How to Define?                                                        |
|:-------------|:--------------------------------------------------------------|:-----------------------------------------------------------------------|
| `setUp()`    | 	**Before** every method whose name starts with `test_` runs. | Define as an **instance method** inside the `unittest.TestCase` class. |
| `tearDown()` | 	**After** every method whose name starts with `test_` runs.  | Define as an **instance method** inside the `unittest.TestCase` class. |

```python
class TestListOperations(unittest.TestCase):

    def setUp(self):
        # Setup for each test
        print("--- METHOD SETUP: Creating a fresh list for the test. ---")
        # Store the resource on the instance itself using 'self'
        self.data = [1, 2, 3]

    def tearDown(self):
        # Cleanup for each test
        print("--- METHOD TEARDOWN: Deleting the list. ---")
        del self.data

    def test_append(self):
        # Test 1: starts with a fresh self.data = [1, 2, 3]
        self.data.append(4)
        self.assertEqual(len(self.data), 4)

    def test_pop(self):
        # Test 2: also starts with a fresh self.data = [1, 2, 3]
        self.data.pop()
        self.assertEqual(self.data, [1, 2])
```

### Creating & running a Test Package

* Generally a large code base project contains multiple modules (*.py files) and hence required multiple test modules.
* It's always good to keep test modules as a unit and separate it from development code.
* A `Test package` is nothing but a python package containing all test modules.

**Create a test module** as follows:

* Assuming your project code is in a folder called `Project1`...
* Create a sub-folder names `test` under `Project1` folder.
* Create an `__init__.py` file inside the `test` sub-folder.
* Creat all your test modules in the `test` folder (e.g. `test_add2nums.py`, `test_pow2num.py` etc.) - as a convention,
  all test modules are named `test_*.py`
* Add expression `all = [test_add2nums, test_pow2num,...]` in `__init__.py` (list all `test_` modules here!)

**Run the test module** as follows:

* From the `Project1` folder fun the following command

```bash
$> python -m unittest discover
```

The above command will do the following:

* **Starts in the Current Directory**: Since you run it from Project1, it sets the starting directory for the search to
  `Project1`.
* **Searches Subdirectories**: It recursively searches all subdirectories (including `src` and `tests`).
* **Looks for Test Files**: It treats _any file_ matching the default pattern, which is `test*.py` (e.g.,
  `test_add2nums.py`), as a test file.
* **Loads Test Cases**: It loads any class derived from `unittest.TestCase` inside those files and runs them.

Alternatively from the `Project1` folder, you can run the following commands:

```bash
$> python -m unittest tests/test_add2nums.py tests/test_pow2nums.py
```

OR

```bash
$> python -m unittest tests.test_add2nums tests.test_pow2nums
```

#### `assert` methods of `unittest.TestCase` class

* The unittest.TestCase class provides a large number of assert* methods for checking specific conditions. They are the
  core tool for making statements about the behavior of your code.
* Here is a list of the most common assertion methods, followed by a complete example demonstrating their use with a
  fictitious class.

| Category   | Method                                                                             | Checks for...                                                                                                                                                                                                                                                                                                                                     |
|:-----------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Equality   | `assertEqual(a, b)`                                                                | $a == b$                                                                                                                                                                                                                                                                                                                                          |
|            | `assertNotEqual(a, b)`                                                             | $a \neq b$                                                                                                                                                                                                                                                                                                                                        |
| Boolean    | `assertTrue(x)`                                                                    | `bool(x)` is `True`                                                                                                                                                                                                                                                                                                                               |
|            | `assertFalse(x)`                                                                   | `bool(x)` is `False`                                                                                                                                                                                                                                                                                                                              |
| Identity   | `assertIs(a, b)`                                                                   | $a$ is $b$ (same object)                                                                                                                                                                                                                                                                                                                          |
|            | `assertIsNot(a, b)`                                                                | $a$ is not $b$ (different object)                                                                                                                                                                                                                                                                                                                 |
| Membership | `assertIn(a, container)`                                                           | $a$ is in container                                                                                                                                                                                                                                                                                                                               |
|            | `assertNotIn(a, container)`                                                        | $a$ is not in container                                                                                                                                                                                                                                                                                                                           |
| Exceptions | `assertRaises(Exc, callable, ...)`                                                 | `callable` raises exception `Exc`                                                                                                                                                                                                                                                                                                                 |
| Comparison | `assertGreater(a, b)`                                                              | $a > b$                                                                                                                                                                                                                                                                                                                                           |
|            | `assertLessEqual(a, b)`                                                            | $a \le b$                                                                                                                                                                                                                                                                                                                                         |
|            | `assertRegex(text, expected_regex)` <br/> `assertNotRegex(text, unexpected_regex)` | Asserts that the expected_regex pattern is found/not-found anywhere within the text string.                                                                                                                                                                                                                                                       |
|            | `assertMultiLineEqual(str1, str2)`                                                 | Compares two strings, treating them as multi-line sequences.<br/> If the strings are not equal, it prints a side-by-side or unified diff output, similar to what version control systems (like Git) produce, highlighting exactly which lines or characters differ. This is significantly more useful than a simple assertEqual for large strings |
| Object Type |`assertIsInstance(obj, cls)`                                                                                                                    | `obj` is an instance of `cls`                                                                                                                                                                                                                                                                                                                     |
|Containers | `assertListEqual(list1, list2)`<br/> `assertTupleEqual(tuple1, tuple2)`<br/>
`assertDictEqual(dict1, dict2)` <br/> `assertSetEqual(set1, set2)` | Verifies that two lists/tuples/dicts/sets are equal and have the same order (order does not apply to sets!). |

```python
# examples 
# Check if the log line contains a timestamp in YYYY-MM-DD format
timestamp_regex = r"\d{4}-\d{2}-\d{2}"

self.assertRegex(log_line, timestamp_regex,
                 "Log line does not contain the required timestamp format.")  # PASS

html_output = "<html>\n<head>Title</head>\n<body>Content</body>\n</html>"
expected_html = "<html>\n<head>Title</head>\n<body>Content</body>\n</html>"

self.assertMultiLineEqual(html_output, expected_html)  # PASS

# If they differed, the output would clearly show the differing lines:
# self.assertMultiLineEqual(html_output, wrong_html)
# --- Expected
# +++ Actual
# @@ -1,4 +1,4 @@
#  <html>
#  <head>Title</head>
# -<body>Content</body>
# +<body>Different Content</body>
#  </html>

```
#### Skipping Tests
It is not advisable to skip any of the written tests. However, in certain circumstances you may prefer to skip a few tests. This can be done using the following methods:
* `unittest.skip`: skips decorated test unconditionally
* `unittest.skipIf`: skips decorated test only if given condition is met (i.e. True)
* `unittest.skipUnless`: skips a test unless given condition is True
* `unittest.expectedFailure`: when you want to mark a test-case (even if it is decorated) as failure, when you know it will fail (say, because of a known bug)
These are specified as decorators on your test-cases (methods), like this
```python
@unittest.skip("reason for skipping")
def test_sample(self):
  ...

@unittest.skipIf(condition, "reason for skipping")
def test_sample2(self):
  ...

@unittest.skipUnless(condition, "reason for skipping")
def test_sample2(self):
  ...

@unittest.skipUnless(condition, "reason for skipping")
def test_sample2(self):
  ...

@unittest.expectedFailure
def test_known_bug_division(self):
    """This test is expected to fail due to the known bug in divide()."""
    # We expect divide(200, 10) to be 20.0, but the function returns 0.
    self.assertEqual(divide(200, 10), 20.0, 
                      "This test fails because the function returns 0 instead of 20.0")
```
