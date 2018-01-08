:: Copyright 2017 IBM RESEARCH. All Rights Reserved.
:: 
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
:: =============================================================================

@ECHO OFF
@SETLOCAL enabledelayedexpansion

pushd %~dp0

SET target=%~n1

IF "%target%"=="env" GOTO :env
IF "%target%"=="run" GOTO :run
IF "%target%"=="lint" GOTO :lint
IF "%target%"=="test" GOTO :test
IF "%target%"=="profile" GOTO :profile
IF "%target%"=="doc" GOTO :doc
IF "%target%"=="clean" GOTO :clean
:usage
ECHO.
ECHO.Usage:
ECHO.    .\make env     Switches to a Python virtual environment
ECHO.    .\make run     Runs Jupyter tutorials
ECHO.    .\make lint    Runs Pyhton source code analisys tool
ECHO.    .\make test    Runs tests
ECHO.    .\make prfile  Runs profiling tests
ECHO.    .\make doc     Creates documentation
ECHO.    .\make clean   Cleans previoulsy generated documentation
ECHO.
GOTO :end

:env
SET QISKIT_ENV_FOUND=No
@FOR /F %%i IN ('conda info --envs') DO (
	IF "%%i"=="QISKitenv" SET QISKIT_ENV_FOUND=Yes
)
IF "%QISKIT_ENV_FOUND%"=="No" (
	conda create -y -n QISKitenv python=3
)
IF errorlevel 9009 GOTO :error
activate QISKitenv & pip install -r requirements.txt
IF errorlevel 9009 GOTO :error
GOTO :next

:run
cd examples\jupyter
jupyter notebook
IF errorlevel 9009 GOTO :error
GOTO :next

:lint
pylint qiskit test
IF errorlevel 9009 GOTO :error
GOTO :next

:test
pip install -r requirements.txt
IF errorlevel 9009 GOTO :error
python -m unittest discover -v
IF errorlevel 9009 GOTO :error
GOTO :next

:profile
python -m unittest discover -p "profile*.py" -v
IF errorlevel 9009 GOTO :error
GOTO :next

:doc
SET PYTHONPATH=$(PWD)
sphinx-apidoc -f -o doc\_autodoc -d 5 -P -e qiskit
IF errorlevel 9009 GOTO :error
cd doc
make.bat html
GOTO :next

:clean
cd doc
make.bat clean
GOTO :next

:error
ECHO.
ECHO.Somehting is missing in your Python installation.
ECHO.Please make sure you have properly installed Anaconda3
ECHO.(https://www.continuum.io/downloads), and that you are
ECHO.running the "Anaconda Shell Command".
ECHO.
exit /b 1

:next

:end
popd
ENDLOCAL