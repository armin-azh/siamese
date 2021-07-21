@echo off


title Deployment
echo Face Recognition
echo System name : %ComputerName%
echo User name : %USERNAME%
echo ----------------------------
setlocal enabledelayedexpansion
SET username=%USERNAME%
set envname=activate.bat
SET posturl=\anaconda3\Scripts\
SET preurl=C:\Users\
SET url= %preurl%%username%%posturl%%envname%
start %url%

pause