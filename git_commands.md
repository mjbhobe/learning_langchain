# Git Commands
This file lists all the git commands you can run to connect a local folder to a remote Github repo.

**Assumptions**:
Assume the following steps have been completed:
- You have created a repo on Github called `Pytorch-Project` (so it will be available at `https://github.com/yourusername/Pytorch-Projects.git`)
- You are developing all your code in a local folder `/dev/code/pytorch_project`

**Steps**:
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
4. **IMP:**Feth the latest files from remote repo
```bash
$> git pull origin main --allow-unrelated-histories
```
The following commands to be run every time you add new files or edit/change existing ones. 
**Run these in the sequence shown**
```bash
$> git add -A  # same as "Stage all changes"
$> git commit -m "<<your comment>>" # same as "Commit Staged"
$> git pull    # pull all remote changes (optional)
$> git push origin main
```



