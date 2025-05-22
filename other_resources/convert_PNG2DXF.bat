@echo off
REM ======= Parameters Setting =======
set INPUT_DIR=..\sample_data\MNIST\spliced_PNG
set OUTPUT_DIR=..\sample_data\MNIST\spliced_DXF
set MAX_IMAGES=1

REM ======= Make Output File =======
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM ======= Initialize Counter =======
setlocal enabledelayedexpansion
set COUNT=0

REM >nul 2>&1

REM ======= Process PNG Files =======
for %%f in (%INPUT_DIR%\*.png) do (
    if !COUNT! geq %MAX_IMAGES% goto DONE

    echo Processing: %%~nxf

    REM PNG → BMP, closed warning message!!!
    magick "%%f" -monochrome "%OUTPUT_DIR%\%%~nf.bmp" 

    REM BMP → SVG
    potrace -s "%OUTPUT_DIR%\%%~nf.bmp" -o "%OUTPUT_DIR%\%%~nf.svg"  

    REM SVG → DXF
    inkscape "%OUTPUT_DIR%\%%~nf.svg" --export-type="dxf" --export-filename="%OUTPUT_DIR%\%%~nf.dxf"  

    set /a COUNT+=1
)

:DONE
echo ✅ Converted %COUNT% PNG to DXF!
pause


