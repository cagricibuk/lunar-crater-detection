package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Step 4 of pipeline: Region-based validation filter.
 *
 * For each Hough circle candidate, checks region properties
 * to reject false positives:
 *   - Circularity score of the edge mask within the circle
 *   - Minimum / maximum area (π·r²)
 *
 * Reference: Region properties — Week 4, COMP4687 syllabus.
 */
public class RegionFilter {

    private double minCircularity = 0.15;  // 0..1, 1 = perfect circle
    private double minAreaRatio   = 0.1;  // min fraction of circle area covered by edges
    private int    minRadius      = 8;
    private int    maxRadius      = 200;

    /**
     * Filter crater candidates against edge image.
     *
     * @param candidates  raw Hough circles: List of int[]{cx, cy, r}
     * @param edges       binary edge image from EdgeDetector
     * @return            validated craters
     */
    public List<int[]> filter(List<int[]> candidates, Mat edges) {
        List<int[]> valid = new ArrayList<>();

        for (int[] c : candidates) {
            int cx = c[0], cy = c[1];
            // Use average radius for region filter proxy
            int r = (c.length >= 5) ? (c[2] + c[3]) / 2 : c[2];

            // 1. Radius range
            if (r < minRadius || r > maxRadius) continue;

            // 2. Bounds check — circle must be fully inside image
            if (cx - r < 0 || cy - r < 0 ||
                cx + r >= edges.cols() || cy + r >= edges.rows()) continue;

            // 3. Circularity of edge pixels inside the ring band
            double circ = computeCircularity(edges, cx, cy, r);
            if (circ < minCircularity) continue;

            valid.add(c);
        }

        return valid;
    }

    /**
     * Measure how "circular" the edge distribution is around (cx,cy,r).
     * We sample the edge pixels in a thin band [r-delta, r+delta] and
     * compute: (edgePixelsInBand) / (expectedCircumference).
     */
    private double computeCircularity(Mat edges, int cx, int cy, int r) {
        int delta = Math.max(2, r / 10);
        int rMin2 = (r - delta) * (r - delta);
        int rMax2 = (r + delta) * (r + delta);

        int edgeCount = 0;
        int bandCount = 0;

        int x0 = Math.max(0, cx - r - delta);
        int x1 = Math.min(edges.cols() - 1, cx + r + delta);
        int y0 = Math.max(0, cy - r - delta);
        int y1 = Math.min(edges.rows() - 1, cy + r + delta);

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                int dx = x - cx, dy = y - cy;
                int dist2 = dx*dx + dy*dy;
                if (dist2 >= rMin2 && dist2 <= rMax2) {
                    bandCount++;
                    double[] px = edges.get(y, x);
                    if (px != null && px[0] > 0) edgeCount++;
                }
            }
        }

        if (bandCount == 0) return 0.0;
        double expectedCircumference = 2 * Math.PI * r;
        return (double) edgeCount / expectedCircumference;
    }

    // --- setters ---
    public void setMinCircularity(double v) { this.minCircularity = v; }
    public void setMinAreaRatio(double v)   { this.minAreaRatio   = v; }
    public void setMinRadius(int v)         { this.minRadius      = v; }
    public void setMaxRadius(int v)         { this.maxRadius      = v; }

    // --- getters ---
    public double getMinCircularity() { return minCircularity; }
    public double getMinAreaRatio()   { return minAreaRatio; }
    public int    getMinRadius()      { return minRadius; }
    public int    getMaxRadius()      { return maxRadius; }
}
