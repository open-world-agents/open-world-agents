@echo off
setlocal enabledelayedexpansion

:: Ensure an argument is provided
if "%~1"=="" (
    echo Usage: %~nx0 ^<arg^>
    pause
    exit /b 1
)

:: Extract the provided argument
set "ARG=%~1"

:: Step 1: Check if env directory exists
if not exist "env\" (
    echo Extracting env.tar.gz...
    
    mkdir env || (
        echo Failed to create env directory
        pause
        exit /b 1
    )

    tar -xvf env.tar.gz -C env || (
        echo Failed to extract env.tar.gz
        pause
        exit /b 1
    )
) else (
    echo env directory already exists. Skipping extraction.
)

:: Step 2: Change directory to env
cd env || (
    echo Failed to enter env directory
    pause
    exit /b 1
)

:: Step 3: Check if virtual environment is activated
if "%CONDA_DEFAULT_ENV%"=="" (
    echo Activating virtual environment...
    call .\Scripts\activate.bat || (
        echo Failed to activate environment
        pause
        exit /b 1
    )
) else (
    echo Virtual environment is already activated.
)

:: Step 4: Check if conda-unpack.exe exists and run it only if needed
if exist ".\Scripts\conda-unpack.exe" (
    echo Running conda-unpack.exe...
    call .\Scripts\conda-unpack.exe || (
        echo Failed to run conda-unpack.exe
        pause
        exit /b 1
    )
) else (
    echo conda-unpack.exe not found, skipping this step.
)

:: Step 5: Run CLI with the argument (preserving colors)
echo Running owl with argument: %ARG%

:: Method 1: Direct execution (best for color support)
@REM recorder.exe "%ARG%"
python -m owa.cli "%ARG%"
set ERR=%errorlevel%

:: Alternative Method 2: Use start /wait if the output is still not fully visible
:: start /wait recorder.exe "%ARG%"
:: set ERR=%errorlevel%

if %ERR% neq 0 (
    echo Failed to run owl with exit code %ERR%
    pause
    exit /b %ERR%
)

echo Script execution completed.
pause  :: Keeps the window open
exit /b 0