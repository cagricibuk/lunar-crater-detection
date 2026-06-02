package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Step 3 of pipeline: Circular Hough Transform.
 *
 * Finds circular crater candidates in edge image.
 * Reference: Hough Transform — Week 8, COMP4687 syllabus.
 */
public class HoughDetector {

    // HoughCircles parameters
    private double dp          = 1.2;   // inverse resolution ratio
    private double minDist     = 20.0;  // min distance between circle centers
    private double param1      = 100.0; // Canny high threshold (internal)
    private double param2      = 30.0;  // accumulator threshold → lower = more circles
    private int    minRadius   = 10;
    private int    maxRadius   = 150;

    /**
     * Detect circles in the preprocessed (gray, smoothed) image.
     * NOTE: HoughCircles works best on the blurred gray image, not the edge image.
     *
     * @param preprocessed  gray, Gaussian-blurred Mat from Preprocessor
     * @return list of int[]{centerX, centerY, radius}
     */
    public List<int[]> detect(Mat preprocessed) {
        Mat circles = new Mat();

        Imgproc.HoughCircles(
            preprocessed,
            circles,
            Imgproc.HOUGH_GRADIENT,
            dp,
            minDist,
            param1,
            param2,
            minRadius,
            maxRadius
        );

        List<int[]> result = new ArrayList<>();
        int maxCandidates = Math.min(circles.cols(), 300);
        for (int i = 0; i < maxCandidates; i++) {
            double[] c = circles.get(0, i);
            if (c != null) {
                int r = (int) Math.round(c[2]);
                result.add(new int[]{
                    (int) Math.round(c[0]),  // cx
                    (int) Math.round(c[1]),  // cy
                    r,                       // a
                    r,                       // b
                    0                        // angle
                });
            }
        }

        circles.release();
        return result;
    }

    // --- setters ---
    public void setDp(double v)        { this.dp        = v; }
    public void setMinDist(double v)   { this.minDist   = v; }
    public void setParam1(double v)    { this.param1    = v; }
    public void setParam2(double v)    { this.param2    = v; }
    public void setMinRadius(int v)    { this.minRadius = v; }
    public void setMaxRadius(int v)    { this.maxRadius = v; }

    // --- getters ---
    public double getDp()        { return dp; }
    public double getMinDist()   { return minDist; }
    public double getParam1()    { return param1; }
    public double getParam2()    { return param2; }
    public int    getMinRadius() { return minRadius; }
    public int    getMaxRadius() { return maxRadius; }
}
