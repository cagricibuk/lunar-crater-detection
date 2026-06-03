package pipeline;

import algorithms.Preprocessor;
import algorithms.KMeansSegmenter;
import algorithms.EdgeDetector;
import algorithms.EllipseDetector;
import algorithms.HoughDetector;
import algorithms.RegionFilter;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * Orchestrates the full crater detection pipeline.
 *
 * Pipeline order:
 *   1. Load image
 *   2. Preprocessor      → CLAHE + Top-Hat + Morph Gradient + Gaussian
 *   2b. KMeansSegmenter  → K=3 topographic masking (shadow isolation)
 *   3. EdgeDetector      → Canny or LoG
 *   4. HoughDetector     → Circular Hough Transform
 *   5. RegionFilter      → circularity validation
 *   6. Draw overlay      → result Mat
 */
public class CraterPipeline {

    private final Preprocessor    preprocessor    = new Preprocessor();
    private final KMeansSegmenter kmeansSegmenter = new KMeansSegmenter();
    private final EdgeDetector    edgeDetector    = new EdgeDetector();
    private final HoughDetector   largeHoughDetector = new HoughDetector();
    private final HoughDetector   smallHoughDetector = new HoughDetector();
    private final EllipseDetector ellipseDetector = new EllipseDetector();
    private final RegionFilter    regionFilter    = new RegionFilter();

    /**
     * Run the full pipeline on an image file.
     */
    public PipelineResult run(String imagePath) {
        return run(imagePath, null);
    }

    /**
     * Run the full pipeline on an image file with progress callback.
     */
    public PipelineResult run(String imagePath, java.util.function.DoubleConsumer progressCallback) {
        Mat original = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_COLOR);
        if (original.empty()) {
            PipelineResult result = new PipelineResult();
            result.appendLog("ERROR: Could not load image: " + imagePath);
            return result;
        }
        PipelineResult res = run(original, progressCallback);
        original.release();
        return res;
    }

    /**
     * Run the full pipeline on an already loaded Mat.
     */
    public PipelineResult run(Mat original) {
        return run(original, null);
    }

    /**
     * Run the full pipeline on an already loaded Mat with progress callback.
     */
    public PipelineResult run(Mat original, java.util.function.DoubleConsumer progressCallback) {
        PipelineResult result = new PipelineResult();
        long startTime = System.currentTimeMillis();

        result.setMat(PipelineResult.Stage.ORIGINAL, original.clone());
        result.appendLog("Processing Mat [" + original.cols() + "x" + original.rows() + "]");

        // 2. Preprocess
        if (progressCallback != null) progressCallback.accept(0.1);
        Mat preprocessed = preprocessor.process(original);
        if (progressCallback != null) progressCallback.accept(0.15);
        result.appendLog("Preprocessing done (CLAHE=" + preprocessor.getClipLimit() +
            ", TopHat=" + preprocessor.getTopHatKernel() +
            ", Morph=" + preprocessor.getMorphKernel() + "x" + preprocessor.getMorphIterations() +
            ", Gaussian=" + preprocessor.getGaussKernel() + ")");

        // 2b. K-Means Segmentation — topographic shadow isolation
        Mat mask = kmeansSegmenter.segment(preprocessed);
        Mat masked = Mat.zeros(preprocessed.size(), preprocessed.type());
        preprocessed.copyTo(masked, mask);   // keep only shadow-region intensities
        if (progressCallback != null) progressCallback.accept(0.25);

        if (kmeansSegmenter.isEnabled()) {
            result.appendLog("K-Means segmentation: " + kmeansSegmenter.getLastSummary());
        } else {
            result.appendLog("K-Means segmentation: DISABLED (pass-through)");
        }
        result.setMat(PipelineResult.Stage.PREPROCESSED, toDisplayable(masked));

        // 3. Edge detection (operates on shadow-masked image)
        Mat edges = edgeDetector.detect(masked);
        if (progressCallback != null) progressCallback.accept(0.4);
        result.setMat(PipelineResult.Stage.EDGES, toDisplayable(edges));
        result.appendLog("Edge detection: " + edgeDetector.getMode() +
            " (t1=" + edgeDetector.getThreshold1() +
            ", t2=" + edgeDetector.getThreshold2() + ")");

        // 4. Hough circles (large and small scale)
        // Pass 'masked' (grayscale) instead of 'edges' so internal Canny can work properly with param1
        List<int[]> largeCircles = largeHoughDetector.detect(masked);
        List<int[]> smallCircles = smallHoughDetector.detect(masked);
        
        List<int[]> houghCircles = new java.util.ArrayList<>(largeCircles);
        houghCircles.addAll(smallCircles);
        
        if (progressCallback != null) progressCallback.accept(0.6);
        result.appendLog("Hough candidates - Large: " + largeCircles.size() +
            ", Small: " + smallCircles.size());

        // 4b. Ellipse detection (on edges)
        List<int[]> ellipses = ellipseDetector.detect(edges);
        if (progressCallback != null) progressCallback.accept(0.7);
        result.appendLog("Ellipse candidates: " + ellipses.size() +
            " (minArc=" + ellipseDetector.getMinArcLength() + ")");

        // Merge & deduplicate
        List<int[]> candidates = new java.util.ArrayList<>(houghCircles);
        for (int[] e : ellipses) {
            boolean isDuplicate = false;
            double eAvgRadius = (e[2] + e[3]) / 2.0;
            double threshold = eAvgRadius * 0.3; // relative merge threshold
            
            for (int[] h : houghCircles) {
                double dist = Math.hypot(e[0] - h[0], e[1] - h[1]);
                if (dist < threshold) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate) candidates.add(e);
        }
        if (progressCallback != null) progressCallback.accept(0.8);
        result.appendLog("Merged candidates: " + candidates.size());

        // 5. Region filter
        List<int[]> craters = regionFilter.filter(candidates, edges);
        if (progressCallback != null) progressCallback.accept(0.9);
        result.appendLog("After region filter: " + craters.size() + " craters");

        // 6. Draw overlay
        Mat detected = original.clone();
        


        for (int[] c : craters) {
            result.addCrater(c);
            
            boolean isCircle = (c[2] == c[3]);
            Scalar color = isCircle ? new Scalar(0, 255, 80) : new Scalar(0, 255, 255); // Green for circle, Yellow for ellipse

            if (isCircle) {
                Imgproc.circle(detected, new Point(c[0], c[1]), c[2], color, 2);
            } else {
                Imgproc.ellipse(detected, new Point(c[0], c[1]), new Size(c[2], c[3]), c[4], 0, 360, color, 2);
            }

            // center dot
            Imgproc.circle(detected, new Point(c[0], c[1]), 3, new Scalar(0, 120, 255), -1);

            // label
            int avgR = (c[2] + c[3]) / 2;
            Imgproc.putText(detected, "r=" + avgR, new Point(c[0] + avgR + 4, c[1]), Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
        result.setMat(PipelineResult.Stage.DETECTED, detected);

        if (progressCallback != null) progressCallback.accept(1.0);

        result.setProcessingTimeMs(System.currentTimeMillis() - startTime);
        result.appendLog("Done in " + result.getProcessingTimeMs() + " ms");

        // release intermediates
        preprocessed.release();
        mask.release();
        masked.release();
        edges.release();

        return result;
    }

    /** Convert single-channel Mat to 3-channel for display */
    private Mat toDisplayable(Mat mat) {
        if (mat.channels() == 1) {
            Mat color = new Mat();
            Imgproc.cvtColor(mat, color, Imgproc.COLOR_GRAY2BGR);
            return color;
        }
        return mat.clone();
    }

    // --- expose algorithm objects so GUI can bind sliders ---
    public Preprocessor    getPreprocessor()      { return preprocessor; }
    public KMeansSegmenter getKmeansSegmenter()   { return kmeansSegmenter; }
    public EdgeDetector    getEdgeDetector()      { return edgeDetector; }
    public HoughDetector   getLargeHoughDetector() { return largeHoughDetector; }
    public HoughDetector   getSmallHoughDetector() { return smallHoughDetector; }
    public EllipseDetector getEllipseDetector()   { return ellipseDetector; }
    public RegionFilter    getRegionFilter()      { return regionFilter; }
}
