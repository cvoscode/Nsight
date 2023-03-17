setlocal

REM Set the name of the virtual environment.
set VENV_NAME=env

REM Check if a Python executable path was passed as an argument.
if "%1"=="" (
    echo Please specify the path to the Python executable as a command-line argument.
    goto end
) else (
    set PYTHON_EXE=%1
)

REM Create the virtual environment.
%PYTHON_EXE% -m venv %VENV_NAME%

REM Activate the virtual environment.
call %VENV_NAME%\Scripts\activate

python.exe -m pip install --upgrade pip
echo installing dependecies

REM Install the packages specified in requirements.txt
pip install -r requirements.txt

echo finished installing dependecies

type ASCIILogo.txt