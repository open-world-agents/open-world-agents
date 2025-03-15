@echo off
setlocal enabledelayedexpansion

:: Configuration
set ENV_DIR=env
set TAR_FILE=env.tar.gz

:: Initialize environment
call :InitializeEnvironment || exit /b 1

:: Run the command
call :RunApplication %* || exit /b 1

echo Script execution completed.
pause
exit /b 0

:: ========= FUNCTIONS =========

:InitializeEnvironment
    :: Extract environment if needed
    if not exist "%ENV_DIR%\" (
        echo Extracting %TAR_FILE%...
        mkdir "%ENV_DIR%" || (
            echo [ERROR] Failed to create %ENV_DIR% directory
            pause
            exit /b 1
        )

        tar -xf %TAR_FILE% -C %ENV_DIR% || (
            echo [ERROR] Failed to extract %TAR_FILE%
            pause
            exit /b 1
        )
    ) else (
        echo %ENV_DIR% directory already exists. Skipping extraction.
    )

    :: Enter environment directory
    cd "%ENV_DIR%" || (
        echo [ERROR] Failed to enter %ENV_DIR% directory
        pause
        exit /b 1
    )

    :: Activate virtual environment if needed
    if "%CONDA_DEFAULT_ENV%"=="" (
        echo Activating virtual environment...
        call .\Scripts\activate.bat || (
            echo [ERROR] Failed to activate environment
            pause
            exit /b 1
        )
    ) else (
        echo Virtual environment is already activated.
    )

    :: Run conda-unpack if available
    if exist ".\Scripts\conda-unpack.exe" (
        echo Running conda-unpack.exe...
        call .\Scripts\conda-unpack.exe || (
            echo [ERROR] Failed to run conda-unpack.exe
            pause
            exit /b 1
        )
    ) else (
        echo conda-unpack.exe not found, skipping this step.
    )

    :: Return to original directory
    cd ..
    exit /b 0

:RunApplication
    echo Running owl with arguments: %*
    python -m owa.cli %*
    set ERR=%errorlevel%
    
    if %ERR% neq 0 (
        echo [ERROR] Failed to run owl with exit code %ERR%
        pause
        exit /b %ERR%
    )
    exit /b 0