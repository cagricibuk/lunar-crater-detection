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



    // Canny parameters
    private double threshold1  = 50.0;
    private double threshold2  = 150.0;
    private int    apertureSize = 3;



    // Morphological Closing parameters
    private int    morphCloseKernel = 5;
    private int    morphCloseIters  = 1;

    /**
     * Detect edges in preprocessed (gray) Mat.
     * Returns binary edge Mat.
     */
    public Mat detect(Mat preprocessed) {
        Mat edges = detectCanny(preprocessed);

        // Apply Morphological Closing to connect broken edges (e.g., incomplete circles)
        Mat closedEdges = new Mat();
        int k = (morphCloseKernel % 2 == 0) ? morphCloseKernel + 1 : morphCloseKernel;
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(k, k));
        Imgproc.morphologyEx(edges, closedEdges, Imgproc.MORPH_CLOSE, element, new Point(-1, -1), morphCloseIters);
        
        edges.release();
        element.release();

        return closedEdges;
    }

    public Mat detectCanny(Mat gray) {
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, threshold1, threshold2, apertureSize, false);
        return edges;
    }



    // --- setters ---
    public void setThreshold1(double v)   { this.threshold1  = v; }
    public void setThreshold2(double v)   { this.threshold2  = v; }
    public void setApertureSize(int v)    { this.apertureSize = v; }
    public void setMorphCloseKernel(int v){ this.morphCloseKernel = (v % 2 == 0) ? v + 1 : v; }
    public void setMorphCloseIters(int v) { this.morphCloseIters = v; }

    // --- getters ---
    public double getThreshold1()   { return threshold1; }
    public double getThreshold2()   { return threshold2; }
    public int    getApertureSize() { return apertureSize; }
    public int    getMorphCloseKernel() { return morphCloseKernel; }
    public int    getMorphCloseIters()  { return morphCloseIters; }
}
