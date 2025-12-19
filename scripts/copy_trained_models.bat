@echo off
REM Copy trained models from runs/train/ to models/trained/

echo Copying trained models...

REM Create target directory
if not exist models\trained mkdir models\trained

REM Copy best.pt for each object
for /d %%D in (runs\train\*) do (
    if exist "%%D\weights\best.pt" (
        for %%F in ("%%D") do set "obj_name=%%~nxF"
        echo   Copying !obj_name!...
        copy "%%D\weights\best.pt" "models\trained\!obj_name!.pt" >nul
    )
)

echo.
echo Done! Trained models copied to models/trained/
dir models\trained\

