# Lunar Crater Detector
### COMP4687 — Introduction to Computer Vision
### Software Development Contest · Işık University

---

## Project Structure

```
LCD/
├── src/
│   ├── Main.java                      ← Entry point
│   ├── ui/
│   │   ├── MainWindow.java            ← JavaFX GUI
│   │   └── MatToImage.java            ← OpenCV Mat → JavaFX Image
│   ├── pipeline/
│   │   ├── CraterPipeline.java        ← Pipeline orchestrator
│   │   └── PipelineResult.java        ← Result data model
│   └── algorithms/
│       ├── Preprocessor.java          ← CLAHE + Gaussian blur
│       ├── EdgeDetector.java          ← Canny / LoG (switchable)
│       ├── HoughDetector.java         ← Circular Hough Transform
│       └── RegionFilter.java          ← Circularity validation
├── lib/
│   ├── opencv.jar                     ← (you add this)
│   └── libopencv_java4xx.so           ← (you add this)
├── resources/
│   └── sample_images/                 ← Put test images here
├── docs/
├── run.sh                             ← Build & run script
└── README.md
```

---

## CV Pipeline (All algorithms from COMP4687 syllabus)

```
Input Image
    │
    ▼
[1] Preprocessor          Week 5-6
    ├─ Grayscale conversion
    ├─ CLAHE (local histogram equalization)   ← handles challenging illumination
    └─ Gaussian blur                          ← noise reduction

    │
    ▼
[2] Edge Detector          Week 7-8
    ├─ Canny edge detector  (default)
    └─ LoG (Laplacian of Gaussian)  (switchable)

    │
    ▼
[3] Hough Detector         Week 8
    └─ Circular Hough Transform (HoughCircles)

    │
    ▼
[4] Region Filter          Week 4
    └─ Circularity score, radius range filter

    │
    ▼
Output: Annotated image + crater list
```

---

## Setup

### 1. Java & JavaFX
- Java 17+ JDK: https://adoptium.net/
- JavaFX SDK 17+: https://gluonhq.com/products/javafx/
  Extract to `~/javafx-sdk/`

### 2. OpenCV Java Bindings

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopencv-dev
find /usr -name "opencv*.jar" 2>/dev/null
find /usr -name "libopencv_java*.so" 2>/dev/null
cp <found.jar> ./lib/opencv.jar
cp <found.so>  ./lib/
```

**macOS:**
```bash
brew install opencv
find /usr/local -name "opencv*.jar"
find /usr/local -name "libopencv_java*.dylib"
```

**Windows:**
- Download OpenCV installer from https://opencv.org/releases/
- After install, find `opencv-4xx.jar` and `opencv_java4xx.dll`
- Copy both to `./lib/`
- In `run.sh`, replace `:` with `;` in the `-cp` argument

### 3. Build & Run
```bash
chmod +x run.sh
./run.sh
```

---

## Test Images

Download from LROC (NASA):
- https://lroc.im-ldi.com/images  (Featured Images)
- Recommended: NAC 90cm images from IM-2 landing region

Put images in `resources/sample_images/`

---

## Citations (per contest rules)

- OpenCV: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
- LROC Images: NASA/GSFC/Arizona State University
- Algorithm references: See COMP4687 lecture slides, Shapiro & Stockman (2001)
