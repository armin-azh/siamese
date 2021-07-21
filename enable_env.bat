echo off
set envname=face-recognition
echo Choose Environment
echo 1) Development
echo 2) Deployment
echo.
CHOICE "Enter your choice:"
IF ERRORLEVEL 1 set FACE_RECOG=dev
IF ERRORLEVEL 2 set FACE_RECOG=dep
echo on
conda activate %envname%