package algorithms;

import org.opencv.core.*;

/**
 * K-Means Segmentation for lunar surface topography isolation.
 *
 * Clusters grayscale pixel intensities into K groups (default K=3):
 *   C1: Deep Shadows  (crater interiors)
 *   C2: Surface        (regolith / general terrain)
 *   C3: Bright Spots   (sun-lit crater rims)
 *
 * Produces a binary mask where the darkest cluster = 255, rest = 0.
 * This isolates crater shadow regions for cleaner edge detection downstream.
 *
 * Implementation: Pure 1D K-Means on intensity values (0–255).
 *   - No OpenCV ML dependency; all math is hand-written.
 *   - Smart initialization (evenly spaced centroids) for fast convergence.
 *   - Typically converges in 5–10 iterations on lunar imagery.
 *
 * Reference: K-Means Clustering — Week 12, COMP4687 syllabus.
 */
public class KMeansSegmenter {

    private int    k             = 3;      // number of clusters
    private int    maxIterations = 20;     // iteration cap
    private double epsilon       = 0.5;    // convergence threshold
    private boolean enabled      = true;   // toggle for pipeline bypass

    // Diagnostic fields (read after segment() call)
    private int    lastIterations = 0;
    private double[] lastCentroids = null;

    /**
     * Run 1D K-Means clustering on pixel intensities.
     *
     * @param gray  single-channel grayscale Mat from Preprocessor
     * @return      binary mask: darkest cluster pixels → 255, rest → 0
     */
    public Mat segment(Mat gray) {
        if (!enabled) {
            // Pass-through: return an all-white mask (no filtering)
            return new Mat(gray.size(), CvType.CV_8UC1, new Scalar(255));
        }

        int rows = gray.rows();
        int cols = gray.cols();
        int totalPixels = rows * cols;

        // ── Step 0: Read all pixel intensities into flat array ──
        byte[] pixels = new byte[totalPixels];
        gray.get(0, 0, pixels);

        // ── Step 1: Smart Initialization ──
        // Evenly spaced centroids: C1=0, C2=127, C3=255 (for K=3)
        double[] centroids = new double[k];
        for (int i = 0; i < k; i++) {
            centroids[i] = (255.0 * i) / (k - 1);
        }

        // Label array: which cluster each pixel belongs to
        int[] labels = new int[totalPixels];

        // ── Iterative Clustering Loop ──
        int iter;
        for (iter = 0; iter < maxIterations; iter++) {

            // ── Step 2: Assignment — assign each pixel to nearest centroid ──
            for (int p = 0; p < totalPixels; p++) {
                int intensity = pixels[p] & 0xFF;   // unsigned byte → int
                double minDist = Double.MAX_VALUE;
                int    bestCluster = 0;

                for (int j = 0; j < k; j++) {
                    // 1D Euclidean distance = absolute difference
                    double dist = Math.abs(intensity - centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }
                labels[p] = bestCluster;
            }

            // ── Step 3: Update — recalculate centroids as arithmetic mean ──
            double[] sums   = new double[k];
            int[]    counts = new int[k];

            for (int p = 0; p < totalPixels; p++) {
                int intensity = pixels[p] & 0xFF;
                sums[labels[p]]   += intensity;
                counts[labels[p]] += 1;
            }

            double maxShift = 0.0;
            for (int j = 0; j < k; j++) {
                if (counts[j] > 0) {
                    double newCentroid = sums[j] / counts[j];
                    maxShift = Math.max(maxShift, Math.abs(newCentroid - centroids[j]));
                    centroids[j] = newCentroid;
                }
            }

            // ── Step 4: Convergence — stop if centroids are stable ──
            if (maxShift < epsilon) {
                iter++;   // count this iteration
                break;
            }
        }

        // Store diagnostics
        lastIterations = iter;
        lastCentroids  = centroids.clone();

        // ── Step 5: Find the darkest cluster (smallest centroid value) ──
        int darkestCluster = 0;
        for (int j = 1; j < k; j++) {
            if (centroids[j] < centroids[darkestCluster]) {
                darkestCluster = j;
            }
        }

        // ── Step 6: Build binary mask ──
        // Darkest cluster → 255 (white), everything else → 0 (black)
        byte[] maskData = new byte[totalPixels];
        for (int p = 0; p < totalPixels; p++) {
            maskData[p] = (labels[p] == darkestCluster) ? (byte) 255 : 0;
        }

        Mat mask = new Mat(rows, cols, CvType.CV_8UC1);
        mask.put(0, 0, maskData);

        return mask;
    }

    // ── Parameter Setters (bound to GUI) ──

    public void setK(int k)              { this.k = Math.max(2, Math.min(k, 10)); }
    public void setMaxIterations(int v)   { this.maxIterations = Math.max(1, v); }
    public void setEpsilon(double v)      { this.epsilon = v; }
    public void setEnabled(boolean v)     { this.enabled = v; }

    // ── Getters ──

    public int     getK()              { return k; }
    public int     getMaxIterations()  { return maxIterations; }
    public double  getEpsilon()        { return epsilon; }
    public boolean isEnabled()         { return enabled; }

    // ── Diagnostics (available after segment() returns) ──

    public int      getLastIterations() { return lastIterations; }
    public double[] getLastCentroids()  { return lastCentroids; }

    /**
     * Returns a human-readable summary of the last clustering result.
     */
    public String getLastSummary() {
        if (lastCentroids == null) return "Not run yet";
        StringBuilder sb = new StringBuilder("K=" + k + ", " + lastIterations + " iters, centroids=[");
        for (int i = 0; i < lastCentroids.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.1f", lastCentroids[i]));
        }
        sb.append("]");
        return sb.toString();
    }
}
