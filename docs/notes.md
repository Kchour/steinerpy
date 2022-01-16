## To create and use virtual environment
```
$ pyenv install 3.8.0
$ cd ~/project_folder
$ pyenv local 3.8.0
$ python3 -m venv my_venv_3.8.0
$ source ~/project_folder/my_venv_3.8.0/bin/activate
```

Note: last commmand is optional when using vscode; 

Then actually install this library 
$ python3 -m pip install -e .

## dependencies
- numpy
- matplotlib