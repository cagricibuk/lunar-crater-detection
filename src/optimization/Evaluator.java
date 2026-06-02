package optimization;

import java.util.List;

public class Evaluator {
    public static class Metrics {
        public int tp = 0, fp = 0, fn = 0;
        
        public double precision() { 
            return (tp + fp == 0) ? 0 : (double) tp / (tp + fp); 
        }
        
        public double recall() { 
            return (tp + fn == 0) ? 0 : (double) tp / (tp + fn); 
        }
        
        public double f1() { 
            double p = precision(), r = recall();
            return (p + r == 0) ? 0 : 2 * p * r / (p + r); 
        }
    }

    /**
     * Evaluates detected craters against ground truth using Bounding Box IoU.
     * Matches are greedy (first one above threshold wins).
     */
    public static Metrics evaluate(List<int[]> detected, List<int[]> truth) {
        Metrics m = new Metrics();
        boolean[] matchedTruth = new boolean[truth.size()];
        
        for (int[] d : detected) {
            int bestTruthIdx = -1;
            double bestIou = 0;
            
            for (int i = 0; i < truth.size(); i++) {
                if (matchedTruth[i]) continue;
                
                double iou = computeIoU(d, truth.get(i));
                if (iou > 0.5 && iou > bestIou) {
                    bestIou = iou;
                    bestTruthIdx = i;
                }
            }
            
            if (bestTruthIdx >= 0) {
                m.tp++;
                matchedTruth[bestTruthIdx] = true;
            } else {
                m.fp++;
            }
        }
        
        for (boolean b : matchedTruth) {
            if (!b) m.fn++;
        }
        
        return m;
    }

    /**
     * Bounding box IoU approximation for circles.
     */
    private static double computeIoU(int[] c1, int[] c2) {
        int x1_min = c1[0] - c1[2], x1_max = c1[0] + c1[2];
        int y1_min = c1[1] - c1[2], y1_max = c1[1] + c1[2];
        
        int x2_min = c2[0] - c2[2], x2_max = c2[0] + c2[2];
        int y2_min = c2[1] - c2[2], y2_max = c2[1] + c2[2];
        
        int i_xmin = Math.max(x1_min, x2_min);
        int i_xmax = Math.min(x1_max, x2_max);
        int i_ymin = Math.max(y1_min, y2_min);
        int i_ymax = Math.min(y1_max, y2_max);
        
        if (i_xmax <= i_xmin || i_ymax <= i_ymin) return 0.0;
        
        double intersection = (i_xmax - i_xmin) * (i_ymax - i_ymin);
        double area1 = (x1_max - x1_min) * (y1_max - y1_min);
        double area2 = (x2_max - x2_min) * (y2_max - y2_min);
        
        return intersection / (area1 + area2 - intersection);
    }
}
