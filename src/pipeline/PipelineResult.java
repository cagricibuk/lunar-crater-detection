package pipeline;

import java.util.List;
import java.util.ArrayList;
import org.opencv.core.Mat;

/**
 * Holds the result of a full crater detection pipeline run.
 */
public class PipelineResult {

    public enum Stage {
        ORIGINAL,
        PREPROCESSED,
        EDGES,
        DETECTED
    }

    // Detected craters: each int[] = {centerX, centerY, radius}
    private List<int[]> craters = new ArrayList<>();

    // Intermediate images for each stage (for side-by-side display)
    private Mat originalMat;
    private Mat preprocessedMat;
    private Mat edgesMat;
    private Mat detectedMat;

    // Stats
    private long processingTimeMs;
    private String log = "";

    // --- Craters ---
    public void addCrater(int[] crater) {
        craters.add(crater);
    }

    public List<int[]> getCraters() { return craters; }
    public int getCraterCount()     { return craters.size(); }

    // --- Mats ---
    public void setMat(Stage stage, Mat mat) {
        switch (stage) {
            case ORIGINAL:     originalMat     = mat; break;
            case PREPROCESSED: preprocessedMat = mat; break;
            case EDGES:        edgesMat        = mat; break;
            case DETECTED:     detectedMat     = mat; break;
        }
    }

    public Mat getMat(Stage stage) {
        switch (stage) {
            case ORIGINAL:     return originalMat;
            case PREPROCESSED: return preprocessedMat;
            case EDGES:        return edgesMat;
            case DETECTED:     return detectedMat;
            default:           return null;
        }
    }

    // --- Stats ---
    public void setProcessingTimeMs(long ms) { this.processingTimeMs = ms; }
    public long getProcessingTimeMs()         { return processingTimeMs; }

    public void appendLog(String line) { this.log += line + "\n"; }
    public String getLog()             { return log; }
}
