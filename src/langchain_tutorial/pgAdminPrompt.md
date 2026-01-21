You are an expert at developing Qt6 Framework based applications using C++ and Python (PyQt). I am interested in developing a complex GUI application that mimics the pgAdmin GUI and which will be used as a SQL client for PostgreSQL, MySQL and SQLite databases.

Need your help to develop 2 sets of code - one in C++ 23 and Qt6 libraries and another with PyQt6. Both sets of code should be spread across multiple files - for C++ version, all class declarations must be in headers & implementations in respective C++ files. For Python versions, code can be spread across multiple modules - all utility functions should be logically grouped into multiple packages. Let's start with the PyQt6 version first - once fully developed, we can convert to a C++ (Qt6) version later. Will prompt you separately for that.

Let's start simple & we'll expand the GUI as we move forward.
1. The main window will be a QFrameWindow derived class, with a menu and a client area that has a vertical QSplitter.

Attached is the typical GUI layout - it consists of 3 parts:
* The treeview on the left, which will list all the "connections" at top level:
    * The very first time the user starts the application, the tree view will be empty (as we have saved no connections to database yet!).
    * under each "connection node" there are nodes names tables, views, stored procedures, triggers. In the next level we have the list of the respective database object - for example, when I click the "tables" node, it should list the names of all the tables in that connection, when I click on the views, it should list names of all "views" and so on.