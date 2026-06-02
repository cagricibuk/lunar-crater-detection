@echo off
setlocal enabledelayedexpansion

:: LunarCraterDetector Windows Build & Run Script

set "JAVAFX_HOME=C:\Users\cagri\javafx-sdk"
set "LIB_DIR=lib"
set "SRC_DIR=src"
set "OUT_DIR=out"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set "JAVAFX_MODS=%JAVAFX_HOME%\lib"
set "OPENCV_JAR=%LIB_DIR%\opencv.jar"
set "NATIVE_LIB_DIR=%LIB_DIR%"

echo === Compiling ===
:: Collect all .java files
dir /s /B "%SRC_DIR%\*.java" > sources.txt
javac --module-path "%JAVAFX_MODS%" --add-modules javafx.controls,javafx.fxml -cp "%OPENCV_JAR%" -d "%OUT_DIR%" @sources.txt
del sources.txt

echo === Running ===
java --module-path "%JAVAFX_MODS%" --add-modules javafx.controls,javafx.fxml -cp "%OUT_DIR%;%OPENCV_JAR%" -Djava.library.path="%NATIVE_LIB_DIR%" Main
