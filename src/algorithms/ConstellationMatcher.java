package algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Implements Geometric Pattern Matching (Constellation Architecture)
 * with Master Crater Pruning, Consensus Voting, and Border Filtering.
 *
 * Pipeline:
 *   1. pruneToMasterCraters: Filters raw Hough output to the most reliable craters
 *      using circularity, minimum radius, and border exclusion (Week 4 Region Props).
 *   2. Selects top-5 template craters → C(5,3) = 10 reference triangles.
 *   3. Searches pruned map triangles for ratio matches + scale constraint.
 *   4. Spatial clustering (Consensus Voting) — multiple triangles pointing to the
 *      same region confirms the real target (Week 12 Pattern Recognition).
 *
 * Syllabus alignment:
 *   - Geometric Vectors & Euclidean Distance  (Week 10)
 *   - Pattern Recognition & Clustering        (Week 12)
 *   - Region Properties & Filtering           (Week 4)
 */
public class ConstellationMatcher {

    // ────────────────────────────────────── Result class

    public static class MatchResult {
        public int x, y;
        public int width, height;
        public double errorScore;    // Lower is better (0.0 = perfect geometric match)
        public double optimalScale;  // Calculated scale (Map / Template)
        public int votes;            // How many triangles voted for this region

        public MatchResult(int x, int y, int width, int height,
                           double errorScore, double optimalScale, int votes) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.errorScore = errorScore;
            this.optimalScale = optimalScale;
            this.votes = votes;
        }
    }

    // ────────────────────────────────────── Triangle helper

    private static class Triangle {
        int[] p1, p2, p3;
        double d0, d1, d2; // Sorted: d0 >= d1 >= d2
        double ratioA;     // d0 / d1
        double ratioB;     // d1 / d2
        double cx, cy;     // Centroid

        public Triangle(int[] p1, int[] p2, int[] p3) {
            this.p1 = p1;
            this.p2 = p2;
            this.p3 = p3;

            double dist1 = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
            double dist2 = Math.hypot(p2[0] - p3[0], p2[1] - p3[1]);
            double dist3 = Math.hypot(p3[0] - p1[0], p3[1] - p1[1]);

            double[] dists = {dist1, dist2, dist3};
            Arrays.sort(dists);
            this.d0 = dists[2];
            this.d1 = dists[1];
            this.d2 = dists[0];

            this.ratioA = (d1 > 0) ? d0 / d1 : Double.MAX_VALUE;
            this.ratioB = (d2 > 0) ? d1 / d2 : Double.MAX_VALUE;

            this.cx = (p1[0] + p2[0] + p3[0]) / 3.0;
            this.cy = (p1[1] + p2[1] + p3[1]) / 3.0;
        }
    }

    // ────────────────────────────────────── Vote accumulator

    private static class VoteCandidate {
        double cx, cy;
        double totalError;
        double scaleSum;
        int votes;
        double bestError;
        double bestScale;
        double bestRefCx, bestRefCy;

        VoteCandidate(double cx, double cy, double error, double scale,
                       double refCx, double refCy) {
            this.cx = cx;
            this.cy = cy;
            this.totalError = error;
            this.scaleSum = scale;
            this.votes = 1;
            this.bestError = error;
            this.bestScale = scale;
            this.bestRefCx = refCx;
            this.bestRefCy = refCy;
        }

        void addVote(double error, double scale, double mapCx, double mapCy,
                     double refCx, double refCy) {
            this.cx = (this.cx * votes + mapCx) / (votes + 1);
            this.cy = (this.cy * votes + mapCy) / (votes + 1);
            this.totalError += error;
            this.scaleSum += scale;
            this.votes++;
            if (error < bestError) {
                bestError = error;
                bestScale = scale;
                bestRefCx = refCx;
                bestRefCy = refCy;
            }
        }

        double averageScale() { return scaleSum / votes; }
        double averageError() { return totalError / votes; }
    }

    // ════════════════════════════════════════════════════════════════════
    //  MASTER CRATER PRUNING (Week 4 — Region Properties)
    // ════════════════════════════════════════════════════════════════════

    /**
     * Filters raw Hough craters down to the most reliable "Master" craters
     * by applying circularity, radius, and border checks.
     *
     * Each crater is int[]{cx, cy, radiusA, radiusB, angle}.
     * Circularity proxy: radiusA / radiusB closeness to 1.0.
     *
     * @param allCraters      Full list from CraterPipeline.
     * @param maxMasterCount  Maximum number of master craters to keep.
     * @param imageWidth      Image width for border exclusion.
     * @param imageHeight     Image height for border exclusion.
     * @param minRadius       Minimum radius to accept (reject noise).
     * @return Pruned list of the most reliable craters.
     */
    public static List<int[]> pruneToMasterCraters(List<int[]> allCraters,
                                                    int maxMasterCount,
                                                    int imageWidth,
                                                    int imageHeight,
                                                    int minRadius) {
        int borderMargin = Math.max(10, (int)(Math.min(imageWidth, imageHeight) * 0.03));

        List<int[]> candidates = new ArrayList<>();
        for (int[] c : allCraters) {
            int cx = c[0], cy = c[1];
            int rA = c[2], rB = (c.length >= 4) ? c[3] : c[2];
            int avgR = (rA + rB) / 2;

            // 1. Minimum radius filter — reject tiny noise
            if (avgR < minRadius) continue;

            // 2. Border exclusion — reject craters too close to image edges
            if (cx - avgR < borderMargin || cy - avgR < borderMargin
                    || cx + avgR > imageWidth - borderMargin
                    || cy + avgR > imageHeight - borderMargin) {
                continue;
            }

            // 3. Circularity proxy — radiusA / radiusB should be close to 1.0
            double circularity = (double) Math.min(rA, rB) / Math.max(rA, rB);
            if (circularity < 0.70) continue; // reject highly elliptical detections

            candidates.add(c);
        }

        // 4. Sort by radius descending (largest = most stable/visible)
        candidates.sort((a, b) -> {
            int rA_a = (a[2] + (a.length >= 4 ? a[3] : a[2])) / 2;
            int rA_b = (b[2] + (b.length >= 4 ? b[3] : b[2])) / 2;
            return Integer.compare(rA_b, rA_a);
        });

        // 5. Take top N
        List<int[]> masters = new ArrayList<>(candidates.subList(0, Math.min(maxMasterCount, candidates.size())));

        System.out.printf("[TRN-PRUNE] %d raw craters → %d candidates → %d masters (minR=%d, border=%dpx)%n",
                allCraters.size(), candidates.size(), masters.size(), minRadius, borderMargin);

        return masters;
    }

    // ════════════════════════════════════════════════════════════════════
    //  MAIN MATCH (Consensus Voting + Scale Constraint)
    // ════════════════════════════════════════════════════════════════════

    /**
     * Finds the template constellation within the map craters using Consensus Voting.
     *
     * @param mapCraters      Pruned master craters from the 50km base map.
     * @param templateCraters Craters detected in the 5km template.
     * @param templateW       Template image width (pixels).
     * @param templateH       Template image height (pixels).
     * @param tolerance       Max allowed ratio error per axis (e.g. 0.05).
     * @param expectedScale   Expected physical scale factor (e.g. 0.10).
     * @return MatchResult with the highest-voted region, or null.
     */
    public static MatchResult match(List<int[]> mapCraters,
                                    List<int[]> templateCraters,
                                    int templateW,
                                    int templateH,
                                    double tolerance,
                                    double expectedScale) {

        if (templateCraters.size() < 3) {
            System.err.println("[TRN-CONST] Error: Not enough craters in template (need >=3, found "
                    + templateCraters.size() + ")");
            return null;
        }
        if (mapCraters.size() < 3) {
            System.err.println("[TRN-CONST] Error: Not enough craters in map (need >=3, found "
                    + mapCraters.size() + ")");
            return null;
        }

        // ── Build multiple reference triangles from top-5 template craters ──
        List<int[]> sortedTemplate = new ArrayList<>(templateCraters);
        sortedTemplate.sort((a, b) -> Integer.compare(b[2], a[2]));

        int topN = Math.min(5, sortedTemplate.size());
        List<int[]> topCraters = sortedTemplate.subList(0, topN);

        List<Triangle> refTriangles = new ArrayList<>();
        for (int i = 0; i < topN - 2; i++)
            for (int j = i + 1; j < topN - 1; j++)
                for (int k = j + 1; k < topN; k++)
                    refTriangles.add(new Triangle(topCraters.get(i),
                                                   topCraters.get(j),
                                                   topCraters.get(k)));

        System.out.printf("[TRN-CONST] Built %d reference triangles from top-%d template craters%n",
                refTriangles.size(), topN);

        // ── Search map and accumulate votes ──
        List<VoteCandidate> candidates = new ArrayList<>();
        double scaleMargin = 0.05;
        // Estimate map extent from crater positions
        int mapW = 0, mapH = 0;
        for (int[] c : mapCraters) {
            if (c[0] + c[2] > mapW) mapW = c[0] + c[2];
            if (c[1] + c[2] > mapH) mapH = c[1] + c[2];
        }
        double clusterRadius = Math.max(mapW, mapH) * 0.05;

        int n = mapCraters.size();
        long possibleTriangles = (long) n * (n - 1) * (n - 2) / 6;
        System.out.printf("[TRN-CONST] Searching %d master craters (%,d possible triangles) x %d ref triangles...%n",
                n, possibleTriangles, refTriangles.size());

        for (Triangle refTri : refTriangles) {
            if (refTri.d0 < 1e-3) continue;

            for (int i = 0; i < n - 2; i++) {
                for (int j = i + 1; j < n - 1; j++) {
                    for (int k = j + 1; k < n; k++) {
                        Triangle mapTri = new Triangle(
                                mapCraters.get(i),
                                mapCraters.get(j),
                                mapCraters.get(k));

                        double errA = Math.abs(refTri.ratioA - mapTri.ratioA);
                        double errB = Math.abs(refTri.ratioB - mapTri.ratioB);

                        if (errA < tolerance && errB < tolerance) {
                            double optimalScale = mapTri.d0 / refTri.d0;

                            // Scale Constraint — reject phantom triangles
                            if (optimalScale >= (expectedScale - scaleMargin)
                                    && optimalScale <= (expectedScale + scaleMargin)) {

                                double totalError = errA + errB;

                                boolean merged = false;
                                for (VoteCandidate vc : candidates) {
                                    double dist = Math.hypot(vc.cx - mapTri.cx, vc.cy - mapTri.cy);
                                    if (dist < clusterRadius) {
                                        vc.addVote(totalError, optimalScale,
                                                mapTri.cx, mapTri.cy,
                                                refTri.cx, refTri.cy);
                                        merged = true;
                                        break;
                                    }
                                }

                                if (!merged) {
                                    candidates.add(new VoteCandidate(
                                            mapTri.cx, mapTri.cy,
                                            totalError, optimalScale,
                                            refTri.cx, refTri.cy));
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Find the candidate with the most votes ──
        VoteCandidate winner = null;
        for (VoteCandidate vc : candidates) {
            System.out.printf("[TRN-CONST]   Candidate at (%.0f, %.0f) — %d votes, avgError=%.4f, avgScale=%.3f%n",
                    vc.cx, vc.cy, vc.votes, vc.averageError(), vc.averageScale());
            if (winner == null || vc.votes > winner.votes
                    || (vc.votes == winner.votes && vc.averageError() < winner.averageError())) {
                winner = vc;
            }
        }

        if (winner == null) {
            System.out.println("[TRN-CONST] ✗ No constellation matched within tolerance.");
            return null;
        }

        // ── Build bounding box from the winner ──
        double finalScale = winner.averageScale();
        int boxX = (int) Math.round(winner.cx - (winner.bestRefCx * finalScale));
        int boxY = (int) Math.round(winner.cy - (winner.bestRefCy * finalScale));
        int boxW = (int) Math.round(templateW * finalScale);
        int boxH = (int) Math.round(templateH * finalScale);

        System.out.printf("[TRN-CONST] ★ Winner: %d votes, avgError=%.4f, scale=%.3f, box=%dx%d at (%d,%d)%n",
                winner.votes, winner.averageError(), finalScale, boxW, boxH, boxX, boxY);

        return new MatchResult(boxX, boxY, boxW, boxH,
                winner.averageError(), finalScale, winner.votes);
    }
}
