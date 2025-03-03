#####################################################################
# This is offered only as a workspace setup, in order to  
# automate fetching the files from the git repository.
# If files are fetched manually, the following shall be ignored.
#####################################################################
import git
import os 

BASE_DIR = 'dantecu'
git.Repo.clone_from('https://github.com/dtecu/kaChallenge', BASE_DIR)
os.chdir(BASE_DIR)
