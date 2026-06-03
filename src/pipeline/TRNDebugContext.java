package pipeline;

import algorithms.TemplateMatcher;
import org.opencv.core.Mat;
import java.awt.image.BufferedImage;

/**
 * Data Transfer Object (DTO) for TRN Pipeline execution (Soft Edge ZNCC mode).
 *
 * Carries the full intermediate state — images and telemetry — 
 * from the background thread to the UI thread for display in the 
 * TRNDiagnosticWindow.
 */
public class TRNDebugContext {

    // ── Input Images ──
    public BufferedImage mapImage;
    public BufferedImage templateImage;

    // ── Intermediate Pipeline Results (Soft Edge Maps) ──
    // Kept as OpenCV Mat for easy display rendering via MatToImage
    public Mat mapSoftEdges;
    public Mat templateSoftEdges;

    // ── Search Telemetry ──
    public long totalZNCCWindowsScanned;
    public double expectedScale;

    // ── Result ──
    public TemplateMatcher.MatchResult finalMatch;
    public long executionTimeMs;
}
