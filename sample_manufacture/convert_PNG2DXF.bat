@echo off
REM ======= Parameters Setting =======
set INPUT_DIR=..\..\data\EMNIST_sample\spliced_PNG_for_testing
set OUTPUT_DIR=..\..\data\EMNIST_sample\spliced_DXF_for_testing
set MAX_IMAGES=10

REM ======= Make Output File =======
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM ======= Initialize Counter =======
setlocal enabledelayedexpansion
set COUNT=0


REM ======= Process PNG Files =======
for %%f in (%INPUT_DIR%\*.png) do (
    if !COUNT! geq %MAX_IMAGES% goto DONE

    echo Processing: %%~nxf

    REM PNG → BMP, closed warning message!!!
    magick "%%f" -monochrome "%OUTPUT_DIR%\%%~nf.bmp" >nul 2>&1

    REM BMP → SVG
    potrace -s "%OUTPUT_DIR%\%%~nf.bmp" -o "%OUTPUT_DIR%\%%~nf.svg" >nul 2>&1  

    REM SVG → DXF
    inkscape "%OUTPUT_DIR%\%%~nf.svg" --export-type="dxf" --export-filename="%OUTPUT_DIR%\%%~nf.dxf" >nul 2>&1

    set /a COUNT+=1
)

:DONE
echo ✅ Converted %COUNT% PNG to DXF!
pause


