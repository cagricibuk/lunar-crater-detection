package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Step 2 of pipeline: Edge detection.
 *
 * Modes (selectable from GUI):
 *   CANNY — Canny edge detector          (Week 7-8, syllabus)
 *   LOG   — Laplacian of Gaussian        (Week 7,   syllabus)
 */
public class EdgeDetector {

    public enum Mode { CANNY, LOG }

    private Mode   mode        = Mode.CANNY;

    // Canny parameters
    private double threshold1  = 50.0;
    private double threshold2  = 150.0;
    private int    apertureSize = 3;

    // LoG parameters
    private int    logKernel   = 9;    // Gaussian kernel size before Laplacian
    private double logSigma    = 2.0;

    /**
     * Detect edges in preprocessed (gray) Mat.
     * Returns binary edge Mat.
     */
    public Mat detect(Mat preprocessed) {
        switch (mode) {
            case LOG:   return detectLoG(preprocessed);
            case CANNY: // fall-through default
            default:    return detectCanny(preprocessed);
        }
    }

    public Mat detectCanny(Mat gray) {
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, threshold1, threshold2, apertureSize, false);
        return edges;
    }

    public Mat detectLoG(Mat gray) {
        // 1. Gaussian blur
        Mat blurred = new Mat();
        int k = (logKernel % 2 == 0) ? logKernel + 1 : logKernel;
        Imgproc.GaussianBlur(gray, blurred, new Size(k, k), logSigma);

        // 2. Laplacian
        Mat laplacian = new Mat();
        Imgproc.Laplacian(blurred, laplacian, CvType.CV_64F);
        blurred.release();

        // 3. Zero-crossing → threshold absolute value
        Mat absLap = new Mat();
        Core.convertScaleAbs(laplacian, absLap);
        laplacian.release();

        Mat edges = new Mat();
        Imgproc.threshold(absLap, edges, 10, 255, Imgproc.THRESH_BINARY);
        absLap.release();

        return edges;
    }

    // --- setters ---
    public void setMode(Mode m)           { this.mode        = m; }
    public void setThreshold1(double v)   { this.threshold1  = v; }
    public void setThreshold2(double v)   { this.threshold2  = v; }
    public void setApertureSize(int v)    { this.apertureSize = v; }
    public void setLogKernel(int v)       { this.logKernel   = v; }
    public void setLogSigma(double v)     { this.logSigma    = v; }

    // --- getters ---
    public Mode   getMode()         { return mode; }
    public double getThreshold1()   { return threshold1; }
    public double getThreshold2()   { return threshold2; }
    public int    getApertureSize() { return apertureSize; }
    public int    getLogKernel()    { return logKernel; }
    public double getLogSigma()     { return logSigma; }
}
