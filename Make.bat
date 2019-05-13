:: Copyright 2017, IBM.
::
:: This source code is licensed under the Apache License, Version 2.0 found in
:: the LICENSE.txt file in the root directory of this source tree.

@ECHO OFF
@SETLOCAL enabledelayedexpansion

pushd %~dp0

SET target=%~n1

IF "%target%"=="env" GOTO :env
IF "%target%"=="run" GOTO :run
IF "%target%"=="lint" GOTO :lint
IF "%target%"=="test" GOTO :test
IF "%target%"=="clean" GOTO :clean
:usage
ECHO.
ECHO.Usage:
ECHO.    .\make env     Switches to a Python virtual environment
ECHO.    .\make run     Runs Jupyter tutorials
ECHO.    .\make lint    Runs Python source code analisys tool
ECHO.    .\make test    Runs tests
ECHO.    .\make prfile  Runs profiling tests
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

:clean
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