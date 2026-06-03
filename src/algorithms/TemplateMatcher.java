package algorithms;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import javax.imageio.ImageIO;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import pipeline.TRNDebugContext;

/**
 * Implements Template Matching using Scale Sweep and ZNCC.
 *
 * Uses the "Soft Edge" (Heavy Gaussian Blur) architecture to overcome the
 * 1-Pixel Edge Trap. Canny edges are blurred to create a "gravity field"
 * around craters, allowing ZNCC to achieve high correlation even with slight
 * scale or viewpoint misalignment.
 */
public class TemplateMatcher {

    public static class MatchResult {
        public int x, y;
        public int width, height;
        public double score;
        public double optimalScale;

        public MatchResult(int x, int y, int width, int height, double score, double optimalScale) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.score = score;
            this.optimalScale = optimalScale;
        }
    }

    public static MatchResult matchWithScaleSweep(BufferedImage mapImage,
                                                   BufferedImage templateImage,
                                                   double targetScale,
                                                   TRNDebugContext ctx) {

        Preprocessor preprocessor = new Preprocessor();
        EdgeDetector edgeDetector = new EdgeDetector();

        // ── Step 1: Map pre-processing (Döngü dışı) ──
        Mat mapMat = bufferedImageToMat(mapImage);
        Mat grayMap = preprocessor.toGray(mapMat);
        Mat claheMap = preprocessor.applyCLAHE(grayMap);
        Mat edgesMap = edgeDetector.detectCanny(claheMap);

        // ── SOFT EDGE YAKLAŞIMI: Dilation yerine Ağır Gaussian Blur ──
        Mat mapSoftEdges = new Mat();
        Imgproc.GaussianBlur(edgesMap, mapSoftEdges, new Size(15, 15), 0);

        // Save map soft edges to context
        if (ctx != null) {
            ctx.mapSoftEdges = mapSoftEdges.clone();
            ctx.totalZNCCWindowsScanned = 0;
        }

        double[][] mapEdgeArray = matToDoubleArray(mapSoftEdges);

        // Debug: save map edge output
        saveDebugImage(mapEdgeArray, "debug_map_soft_edges.png");

        // ── Step 2: Build scale pyramid ──
        double[] scales = {
            targetScale - 0.02,
            targetScale - 0.01,
            targetScale,
            targetScale + 0.01,
            targetScale + 0.02
        };

        MatchResult bestOverall = new MatchResult(0, 0, 0, 0, -1.0, targetScale);

        // ── Step 3: Scale sweep loop ──
        for (double scale : scales) {
            if (scale <= 0.005) continue;

            // 1. Küçült (Bi-linear Interpolation)
            int newW = (int) Math.max(1, Math.round(templateImage.getWidth()  * scale));
            int newH = (int) Math.max(1, Math.round(templateImage.getHeight() * scale));

            BufferedImage scaledTpl = new BufferedImage(newW, newH, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g2d = scaledTpl.createGraphics();
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g2d.drawImage(templateImage, 0, 0, newW, newH, null);
            g2d.dispose();

            Mat scaledMat = bufferedImageToMat(scaledTpl);

            // 2 ve 3. Grayscale ve CLAHE
            Mat grayTemplate = preprocessor.toGray(scaledMat);
            Mat claheTemplate = preprocessor.applyCLAHE(grayTemplate);

            // 4. Canny Edge
            Mat edgesTemplate = edgeDetector.detectCanny(claheTemplate);

            // 5. ── SOFT EDGE YAKLAŞIMI Şablon için ──
            Mat templateSoftEdges = new Mat();
            Imgproc.GaussianBlur(edgesTemplate, templateSoftEdges, new Size(15, 15), 0);

            double[][] tplEdgeArray = matToDoubleArray(templateSoftEdges);

            if (Math.abs(scale - targetScale) < 0.001) {
                saveDebugImage(tplEdgeArray, "debug_template_soft_edges.png");
            }

            int mapW  = mapEdgeArray[0].length;
            int mapH  = mapEdgeArray.length;
            int tplW  = tplEdgeArray[0].length;
            int tplH  = tplEdgeArray.length;

            if (tplW >= mapW || tplH >= mapH) {
                System.out.printf("[TRN-ZNCC] Skipping scale %.3f — template (%dx%d) >= map (%dx%d)%n", scale, tplW, tplH, mapW, mapH);
                continue;
            }

            // Keep track of total combinations
            if (ctx != null) {
                ctx.totalZNCCWindowsScanned += (long)(mapW - tplW + 1) * (mapH - tplH + 1);
            }

            // ZNCC on soft edge maps
            MatchResult scaleResult = runZNCC(mapEdgeArray, tplEdgeArray, scale);

            if (scaleResult != null && scaleResult.score > bestOverall.score) {
                bestOverall = scaleResult;
                if (ctx != null) {
                    if (ctx.templateSoftEdges != null) ctx.templateSoftEdges.release();
                    ctx.templateSoftEdges = templateSoftEdges.clone();
                }
            }

            System.out.printf("[TRN-ZNCC] Scale %.3f → best NCC = %.4f%n", scale, scaleResult != null ? scaleResult.score : -1.0);

            // Memory cleanup for OpenCV Mats
            scaledMat.release();
            grayTemplate.release();
            claheTemplate.release();
            edgesTemplate.release();
            templateSoftEdges.release();
        }

        System.out.printf("[TRN-ZNCC] ★ Winner: scale=%.3f  score=%.4f  pos=(%d,%d)%n",
                          bestOverall.optimalScale, bestOverall.score,
                          bestOverall.x, bestOverall.y);

        // Memory cleanup
        mapMat.release();
        grayMap.release();
        claheMap.release();
        edgesMap.release();
        mapSoftEdges.release();

        if (ctx != null) {
            ctx.finalMatch = bestOverall;
        }

        return bestOverall;
    }

    private static MatchResult runZNCC(double[][] map, double[][] tpl, double scale) {
        int mapW = map[0].length;
        int mapH = map.length;
        int tplW = tpl[0].length;
        int tplH = tpl.length;
        int n    = tplW * tplH;

        double tMean = 0;
        for (int y = 0; y < tplH; y++)
            for (int x = 0; x < tplW; x++)
                tMean += tpl[y][x];
        tMean /= n;

        double tSqSum = 0;
        for (int y = 0; y < tplH; y++)
            for (int x = 0; x < tplW; x++) {
                double d = tpl[y][x] - tMean;
                tSqSum += d * d;
            }

        if (tSqSum < 1e-6) return null;

        double maxScore = -1.0;
        int bestX = 0, bestY = 0;

        for (int y = 0; y <= mapH - tplH; y++) {
            for (int x = 0; x <= mapW - tplW; x++) {
                double iMean = 0;
                for (int ty = 0; ty < tplH; ty++)
                    for (int tx = 0; tx < tplW; tx++)
                        iMean += map[y + ty][x + tx];
                iMean /= n;

                double iSqSum  = 0;
                double crossSum = 0;
                for (int ty = 0; ty < tplH; ty++)
                    for (int tx = 0; tx < tplW; tx++) {
                        double tDiff = tpl[ty][tx]          - tMean;
                        double iDiff = map[y + ty][x + tx]  - iMean;
                        iSqSum  += iDiff * iDiff;
                        crossSum += tDiff * iDiff;
                    }

                double denom = Math.sqrt(tSqSum * iSqSum);
                double ncc = (denom < 1e-6) ? 0.0 : crossSum / denom;

                if (ncc > maxScore) {
                    maxScore = ncc;
                    bestX = x;
                    bestY = y;
                }
            }
        }

        return new MatchResult(bestX, bestY, tplW, tplH, maxScore, scale);
    }

    // --- Helpers to bridge OpenCV Mat and Pure Java ---

    private static Mat bufferedImageToMat(BufferedImage bi) {
        BufferedImage convertedImg = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        convertedImg.getGraphics().drawImage(bi, 0, 0, null);
        byte[] data = ((DataBufferByte) convertedImg.getRaster().getDataBuffer()).getData();
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, data);
        return mat;
    }

    private static double[][] matToDoubleArray(Mat mat) {
        int w = mat.cols();
        int h = mat.rows();
        double[][] array = new double[h][w];
        byte[] buffer = new byte[w * h];
        mat.get(0, 0, buffer);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                array[y][x] = buffer[y * w + x] & 0xFF;
            }
        }
        return array;
    }

    private static void saveDebugImage(double[][] data, String filename) {
        try {
            int h = data.length, w = data[0].length;
            double max = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (data[y][x] > max) max = data[y][x];

            if (max < 1e-6) max = 1.0;

            BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int v = (int) Math.min(255, (data[y][x] / max) * 255);
                    int gray = (v << 16) | (v << 8) | v;
                    img.setRGB(x, y, gray);
                }

            ImageIO.write(img, "png", new File(filename));
            System.out.println("[TRN-DEBUG] Saved: " + new File(filename).getAbsolutePath());
        } catch (Exception e) {
            System.err.println("[TRN-DEBUG] Failed to save " + filename + ": " + e.getMessage());
        }
    }
}
