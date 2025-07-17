@echo off
setlocal enabledelayedexpansion

:: Configuration
set ENV_DIR=env
set TAR_FILE=env.tar.gz

:: Initialize environment
call :InitializeEnvironment || exit /b 1

:: Run the application with auto-detection
call :RunApplication %* || exit /b 1

echo Script execution completed.
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

    :: Restore `ocap` command
    echo Restoring `ocap` command...
    python restore_ocap.py

    exit /b 0

:RunApplication
    :: Auto-detect whether to run command or open shell
    if "%~1"=="" (
        call :OpenInteractiveShell
    ) else (
        call :ExecuteOCAPCommand %*
    )
    exit /b %errorlevel%

:ExecuteOCAPCommand
    echo Running ocap with arguments: %*
    python -m owa.cli %*
    set ERR=%errorlevel%
    
    if %ERR% neq 0 (
        echo [ERROR] Failed to run ocap with exit code %ERR%
        exit /b %ERR%
    )
    exit /b 0

:OpenInteractiveShell
    echo Starting new command prompt with activated environment...
    echo Type 'exit' to close the window when finished.
    echo.
    start cmd.exe /k "call .\env\Scripts\activate.bat && title Conda Environment (%ENV_DIR%)"
    exit /b 0