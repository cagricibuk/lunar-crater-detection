package optimization;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class YoloParser {
    /**
     * Parses a YOLO format annotation file into a list of craters.
     * YOLO format: class_id x_center y_center width height (normalized 0 to 1)
     * Output: List of int[]{cx, cy, radius_in_pixels}
     */
    public static List<int[]> parse(File txtFile, int imgWidth, int imgHeight) throws Exception {
        List<int[]> craters = new ArrayList<>();
        if (!txtFile.exists()) return craters;
        
        List<String> lines = Files.readAllLines(txtFile.toPath());
        for (String line : lines) {
            String[] parts = line.trim().split("\\s+");
            if (parts.length >= 5) {
                double x_norm = Double.parseDouble(parts[1]);
                double y_norm = Double.parseDouble(parts[2]);
                double w_norm = Double.parseDouble(parts[3]);
                double h_norm = Double.parseDouble(parts[4]);
                
                int cx = (int) Math.round(x_norm * imgWidth);
                int cy = (int) Math.round(y_norm * imgHeight);
                int w = (int) Math.round(w_norm * imgWidth);
                int h = (int) Math.round(h_norm * imgHeight);
                
                // For craters, radius is approximately half the average of width and height
                int r = (w + h) / 4; 
                
                craters.add(new int[]{cx, cy, r});
            }
        }
        return craters;
    }
}
