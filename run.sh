#!/usr/bin/env bash
# ============================================================
#  LunarCraterDetector — Build & Run Script
#  COMP4687 · Computer Vision · Işık University
# ============================================================
#
# PREREQUISITES:
#   1. Java 17+ (JDK, not just JRE)
#   2. JavaFX SDK 17+
#      Download: https://gluonhq.com/products/javafx/
#      Extract to:  ~/javafx-sdk/
#
#   3. OpenCV 4.x with Java bindings
#      Download: https://opencv.org/releases/
#      After build/install, you need:
#        - opencv-4xx.jar         → copy to  ./lib/opencv.jar
#        - libopencv_java4xx.so   → copy to  ./lib/   (Linux)
#          opencv_java4xx.dll     → copy to  ./lib/   (Windows)
#          libopencv_java4xx.dylib→ copy to  ./lib/   (macOS)
#
# QUICK OPENCV SETUP (Linux):
#   sudo apt-get install libopencv-dev
#   find /usr -name "opencv*.jar" 2>/dev/null   # find the jar
#   find /usr -name "libopencv_java*.so" 2>/dev/null   # find native lib
#   cp <found.jar>  ./lib/opencv.jar
#   cp <found.so>   ./lib/
#
# ============================================================

set -e

# --- paths (edit these) ---
JAVAFX_HOME="${HOME}/javafx-sdk"          # JavaFX SDK root
LIB_DIR="./lib"
SRC_DIR="./src"
OUT_DIR="./out"

mkdir -p "${OUT_DIR}"

JAVAFX_MODS="${JAVAFX_HOME}/lib"
OPENCV_JAR="${LIB_DIR}/opencv.jar"

# native lib directory for -Djava.library.path
NATIVE_LIB_DIR="${LIB_DIR}"

echo "=== Compiling ==="
javac \
  --module-path "${JAVAFX_MODS}" \
  --add-modules javafx.controls,javafx.fxml \
  -cp "${OPENCV_JAR}" \
  -d "${OUT_DIR}" \
  $(find "${SRC_DIR}" -name "*.java")

echo "=== Running ==="
java \
  --module-path "${JAVAFX_MODS}" \
  --add-modules javafx.controls,javafx.fxml \
  -cp "${OUT_DIR}:${OPENCV_JAR}" \
  -Djava.library.path="${NATIVE_LIB_DIR}" \
  Main

# Windows note: replace ':' with ';' in -cp on Windows CMD
