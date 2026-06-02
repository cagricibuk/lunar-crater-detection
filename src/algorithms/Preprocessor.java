package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.CLAHE;

/**
 * Step 1 of pipeline: Preprocessing for challenging illumination.
 *
 * Techniques used (all from COMP4687 syllabus):
 *   - Grayscale conversion         (Week 4)
 *   - Histogram equalization/CLAHE (Week 5)
 *   - Gaussian smoothing           (Week 6)
 */
public class Preprocessor {

    // CLAHE parameters
    private double clipLimit   = 2.0;   // contrast limit per tile
    private int    tileSize    = 8;     // grid size (NxN tiles)

    // Morphological Gradient parameters
    private int    morphKernel     = 3;     // Kernel size for morph gradient
    private int    morphIterations = 1;

    // White Top-Hat parameters
    private int    topHatKernel    = 15;    // Kernel size for top-hat transform

    // Gaussian blur parameters
    private int    gaussKernel = 5;     // must be odd
    private double gaussSigma  = 1.5;

    /**
     * Run full preprocessing chain on input Mat.
     * Input:  any (color or gray) Mat
     * Output: preprocessed grayscale Mat, ready for edge detection
     */
    public Mat process(Mat input) {
        Mat gray = toGray(input);
        Mat clahe = applyCLAHE(gray);
        Mat tophat = applyTopHat(clahe);
        Mat morph = applyMorphGradient(tophat);
        Mat blurred = applyGaussian(morph);

        // release intermediates we no longer need
        gray.release();
        clahe.release();
        tophat.release();
        morph.release();

        return blurred;
    }

    // --- individual steps (public for step-by-step display) ---

    public Mat toGray(Mat input) {
        Mat gray = new Mat();
        if (input.channels() == 1) {
            input.copyTo(gray);
        } else {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        }
        return gray;
    }

    public Mat applyCLAHE(Mat gray) {
        Mat result = new Mat();
        CLAHE clahe = Imgproc.createCLAHE(clipLimit, new Size(tileSize, tileSize));
        clahe.apply(gray, result);
        return result;
    }

    public Mat applyTopHat(Mat img) {
        Mat result = new Mat();
        int k = (topHatKernel % 2 == 0) ? topHatKernel + 1 : topHatKernel;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(k, k));
        Imgproc.morphologyEx(img, result, Imgproc.MORPH_TOPHAT, element);
        element.release();
        return result;
    }

    public Mat applyMorphGradient(Mat img) {
        Mat result = new Mat();
        int k = (morphKernel % 2 == 0) ? morphKernel + 1 : morphKernel;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(k, k));
        Imgproc.morphologyEx(img, result, Imgproc.MORPH_GRADIENT, element, new Point(-1, -1), morphIterations);
        element.release();
        return result;
    }

    public Mat applyGaussian(Mat gray) {
        Mat result = new Mat();
        int k = (gaussKernel % 2 == 0) ? gaussKernel + 1 : gaussKernel; // ensure odd
        Imgproc.GaussianBlur(gray, result, new Size(k, k), gaussSigma);
        return result;
    }

    // --- parameter setters (bound to GUI sliders) ---

    public void setClipLimit(double v)   { this.clipLimit   = v; }
    public void setTileSize(int v)       { this.tileSize    = Math.max(2, v); }
    public void setTopHatKernel(int v)   { this.topHatKernel = (v % 2 == 0) ? v + 1 : v; }
    public void setMorphKernel(int v)    { this.morphKernel = (v % 2 == 0) ? v + 1 : v; }
    public void setMorphIterations(int v){ this.morphIterations = v; }
    public void setGaussKernel(int v)    { this.gaussKernel = (v % 2 == 0) ? v + 1 : v; }
    public void setGaussSigma(double v)  { this.gaussSigma  = v; }

    // --- getters ---
    public double getClipLimit()  { return clipLimit; }
    public int    getTileSize()   { return tileSize; }
    public int    getTopHatKernel(){ return topHatKernel; }
    public int    getMorphKernel(){ return morphKernel; }
    public int    getMorphIterations(){ return morphIterations; }
    public int    getGaussKernel(){ return gaussKernel; }
    public double getGaussSigma() { return gaussSigma; }
}
