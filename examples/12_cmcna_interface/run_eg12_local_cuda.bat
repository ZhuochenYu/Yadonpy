@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fI"

call D:\Anaconda\condabin\conda.bat activate yadonpy
if errorlevel 1 exit /b 1

set "PYTHONPATH=%REPO_ROOT%\src"
set "YADONPY_GMX_CMD=gmx"
set "YADONPY_CPU_CAP=12"
set "YADONPY_MPI=1"
set "YADONPY_OMP=12"
if not defined YADONPY_GPU set "YADONPY_GPU=1"
if not defined YADONPY_GPU_ID set "YADONPY_GPU_ID=0"
if not defined YADONPY_RESTART set "YADONPY_RESTART=1"
if not defined YADONPY_EG12_TERM_QM set "YADONPY_EG12_TERM_QM=0"

if "%~1"=="" (
    python "%SCRIPT_DIR%run_cmcna_interface.py" --profile smoke
) else (
    python "%SCRIPT_DIR%run_cmcna_interface.py" %*
)

endlocal
