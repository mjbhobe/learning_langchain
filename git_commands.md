# Git and Version Control

## Installation
* Windows download link: [https://git-for-windows.github.io](https://git-for-windows.github.io)
* **Mac installation**: The `xcode-select --install` command run from your terminal will install Git, CLang and other utilities.
* **Linux Installation**
```bash
# for Debian based distros (such as Ubuntu)
$> sudo apt-get update
$> sudo apt-get install git

# for Arch based distros (such as Manjaro KDE)
$> sudo pacman -Syu
$> sudo pacman -S git
```
* After successful install, you **must** configure your user-name and email-ID
```bash
$> git config --global user.name "First Last"
$> git config --global user.email "Email@domain.com"
```

## Some basic terms
* **Git Repository**: A local directory with a `.git` folder where all the contents are tracked for changes.
* **Remote*: A server where all the code repos are saved (for example `GitHub` or `Gitlab`)
* **Commit**: Similar to a version. When you make changes in a file, you need to **commit the changes** in

## Linking Local Folder to Remote Git Repo
This file lists all the git commands you can run to connect a local folder to a remote Github repo.

**Assumptions**:
Assume the following steps have been completed:
- You have created a repo on Github called `Pytorch-Project` (so it will be available at `https://github.com/<<your_username>>/Pytorch-Projects.git`). Your main branch is called _main_
- Assume that you are developing all your code in a local folder `~/dev/code/pytorch_project` (or `c:\users\<<your_user_id>>\dev\code\pytorch_project`) and you did this _after_ creating the repo on Github
- Now you want to _connect_ this _local_ folder to the Github _remote repo_ and let Git manage your code versions going forwared.

### Steps
Run the following commands _in the sequence mentioned below_:
1. Navigate to your local folder<br/>
```bash
$> cd ~/dev/code/pytorch_project   (on Linux/Mac)
OR
$> cd c:\users\<<your_user_id>>\dev\code\pytorch_project   (on Windows)
```
2. Initialize `git` in your local folder as follows:
```bash
$> git init
```
3. Connect local folder to the remote git repo:
```bash
$> git remote add origin https://github.com/<<your_username>>/Pytorch-Projects.git
```
4. **IMPORTANT:** Fetch the latest files from remote repository!
```bash
$> git pull origin main --allow-unrelated-histories
```
**NOTE:** if your main branch was called something other than _main_ (for example _master_), then replace `main` with `master` in the last command above.

That's it - this will _connect_ your _local_ folder to the _remote Git repo_!

### Versioning local files
The following commands to be run every time you add new files or edit/change existing ones. 

**Run these commands from command line in the sequence shown**. As before, the `$>` represents the command prompt & should NOT be typed. The line beginning with `#` is a comment explaining what the _previous_ command does and should NOT be typed.

```bash
$> git add -A  
# same as "Stage all changes" in VS Code

$> git commit -m "<<your commit comment>>" 
# same as "Commit Staged" in VS Code (replace <<your commit comments>> with appropriate commit comments)

$> git pull  OR git pull origin main  
# pull all remote changes (optional)

$> git push origin main 
# push all committed changes to remote repo
```

**NOTE:** if your main branch was called something other than _main_ (for example _master_), then replace `main` with `master` in the last command above.

## Some common Git Commands
| Command       | Explanation              |
|:--------------:|:------------------------|
|`git clone <git_repo_url>` | Get the complete project from remote Git URL to local machine |
|`git pull origin <branch_name>` | Get the new changes from remote branch to local branch |
|`git push origin <branch_name>` | Send your local branch changes to remote branch |
|`git remote add <name> <git_repo_url>` | Add a new remote repo link to your local repo |
|`git remote -v` | List all the remote repo URLs linked to your local repo |