# Git Commands
This file lists all the git commands you can run to connect a local folder to a remote Github repo.

**Assumptions**:
Assume the following steps have been completed:
- You have created a repo on Github called `Pytorch-Project` (so it will be available at `https://github.com/yourusername/Pytorch-Projects.git`). Your main branch is called _main_
- Assume that you are developing all your code in a local folder `/dev/code/pytorch_project` and you did this _after_ creating the rep on Github
- Now you want to _connect_ this _local_ folder to the Github _remote repo_.


### Steps
Run the following commands in sequence:
1. Navigate to your local folder<br/>
```bash
$> cd /dev/code/pytorch_project
```
2. Initialize `git` in your local folder as follows:
```bash
$> git init
```
3. Connect local folder to the remote git repo:
```bash
$> git remote add origin https://github.com/yourusername/Pytorch-Projects.git
```
4. **IMPORTANT:** Fetch the latest files from remote repository!
```bash
$> git pull origin main --allow-unrelated-histories
```
That's it - this will _connect_ your _local_ folder to the _remote Git repo_!

### Versioning local files
The following commands to be run every time you add new files or edit/change existing ones. 

**Run these commands from command line in the sequence shown**. As before, the `$>` represents the command prompt & should NOT be typed. The line beginning with `#` is a comment explaining what the _previous_ command does and should NOT be typed.

```bash
$> git add -A  
# same as "Stage all changes" in VS Code

$> git commit -m "<<your comment>>" 
# same as "Commit Staged" in VS Code (replace <<your comments>> with appropriate commit comments)

$> git pull    
# pull all remote changes (optional)

$> git push origin main 
# push all committed changes to remote repo
```
**NOTE:** if your main branch was called something other than _main_ (for example _master_), then replace `main` with `master` in the last command above.


