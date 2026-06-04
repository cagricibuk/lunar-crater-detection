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
 *   - Quadrant symmetry: edge distribution across 4 quadrants
 *
 * Reference: Region properties — Week 4, COMP4687 syllabus.
 */
public class RegionFilter {

    private double minCircularity   = 0.15;  // 0..1, 1 = perfect circle
    private int    minRadius        = 8;
    private int    maxRadius        = 200;
    private double maxQuadrantRatio = 3.0;   // max imbalance between opposing quadrant halves

    /**
     * Filter crater candidates against edge image.
     *
     * @param candidates  raw Hough circles: List of int[]{cx, cy, a, b, angle}
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

            // 4. Quadrant symmetry — reject one-sided edge accumulations
            if (!checkQuadrantSymmetry(edges, cx, cy, r)) continue;

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

    /**
     * Quadrant Symmetry Check — anti false-positive filter for large circles.
     *
     * A real crater rim has edge pixels distributed around the full 360°.
     * False positives (terrain edges, shadow boundaries) accumulate edges
     * on one side only, forming a "(" or ")" shape.
     *
     * We split the ring band into 4 quadrants:
     *   Q1: Top-Right  (dx >= 0, dy < 0)
     *   Q2: Top-Left   (dx < 0,  dy < 0)
     *   Q3: Bottom-Left (dx < 0,  dy >= 0)
     *   Q4: Bottom-Right(dx >= 0, dy >= 0)
     *
     * Then compare opposing halves:
     *   Top (Q1+Q2) vs Bottom (Q3+Q4)
     *   Left (Q2+Q3) vs Right (Q1+Q4)
     *
     * If one half has more than maxQuadrantRatio× the other → reject.
     */
    private boolean checkQuadrantSymmetry(Mat edges, int cx, int cy, int r) {
        int delta = Math.max(2, r / 10);
        int rMin2 = (r - delta) * (r - delta);
        int rMax2 = (r + delta) * (r + delta);

        int[] quadrant = new int[4]; // Q1, Q2, Q3, Q4 edge counts

        int x0 = Math.max(0, cx - r - delta);
        int x1 = Math.min(edges.cols() - 1, cx + r + delta);
        int y0 = Math.max(0, cy - r - delta);
        int y1 = Math.min(edges.rows() - 1, cy + r + delta);

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                int dx = x - cx, dy = y - cy;
                int dist2 = dx*dx + dy*dy;
                if (dist2 >= rMin2 && dist2 <= rMax2) {
                    double[] px = edges.get(y, x);
                    if (px != null && px[0] > 0) {
                        if (dx >= 0 && dy < 0)       quadrant[0]++;  // Q1: Top-Right
                        else if (dx < 0 && dy < 0)   quadrant[1]++;  // Q2: Top-Left
                        else if (dx < 0 && dy >= 0)  quadrant[2]++;  // Q3: Bottom-Left
                        else                          quadrant[3]++;  // Q4: Bottom-Right
                    }
                }
            }
        }

        // Compare opposing halves
        int top    = quadrant[0] + quadrant[1];
        int bottom = quadrant[2] + quadrant[3];
        int left   = quadrant[1] + quadrant[2];
        int right  = quadrant[0] + quadrant[3];

        // Avoid division by zero: if any half is 0, it's definitely one-sided
        int minEdgesForCheck = 3;
        if (top < minEdgesForCheck && bottom < minEdgesForCheck) return false;
        if (left < minEdgesForCheck && right < minEdgesForCheck) return false;

        // Check top-bottom balance
        if (top > 0 && bottom > 0) {
            double tbRatio = (double) Math.max(top, bottom) / Math.min(top, bottom);
            if (tbRatio > maxQuadrantRatio) return false;
        } else {
            // One half is empty → completely one-sided
            return false;
        }

        // Check left-right balance
        if (left > 0 && right > 0) {
            double lrRatio = (double) Math.max(left, right) / Math.min(left, right);
            if (lrRatio > maxQuadrantRatio) return false;
        } else {
            return false;
        }

        return true;
    }

    // --- setters ---
    public void setMinCircularity(double v)   { this.minCircularity   = v; }
    public void setMinRadius(int v)            { this.minRadius        = v; }
    public void setMaxRadius(int v)            { this.maxRadius        = v; }
    public void setMaxQuadrantRatio(double v)  { this.maxQuadrantRatio = Math.max(1.1, v); }

    // --- getters ---
    public double getMinCircularity()   { return minCircularity; }
    public int    getMinRadius()        { return minRadius; }
    public int    getMaxRadius()        { return maxRadius; }
    public double getMaxQuadrantRatio() { return maxQuadrantRatio; }
}
