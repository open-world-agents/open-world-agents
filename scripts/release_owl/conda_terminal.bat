@echo off
setlocal enabledelayedexpansion

:: Configuration
set ENV_DIR=env
set TAR_FILE=env.tar.gz

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

:: Run conda-unpack if available
if exist ".\Scripts\conda-unpack.exe" (
    echo Running conda-unpack.exe...
    call .\Scripts\conda-unpack.exe || (
        echo [ERROR] Failed to run conda-unpack.exe
        pause
        exit /b 1
    )
)

:: Return to original directory
cd ..

:: Restore `owl` command
python restore_owl.py

:: Start a new CMD with the environment activated
echo Starting new command prompt with activated environment...
echo Type 'exit' to close the window when finished.
echo.
start cmd.exe /k "call .\env\Scripts\activate.bat && title Conda Environment (%ENV_DIR%)"

:: Exit this script
exit /b 0