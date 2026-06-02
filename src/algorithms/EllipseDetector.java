package algorithms;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Step 3b of pipeline: Contour-based Ellipse Detector.
 *
 * Finds non-circular (elliptical) crater candidates by finding
 * contours in the edge image and fitting ellipses to them.
 */
public class EllipseDetector {

    private double minArcLength = 30.0;
    private double minAspectRatio = 0.5;

    public List<int[]> detect(Mat edges) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        
        // Find contours on the binary edge image
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        
        List<int[]> result = new ArrayList<>();
        
        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            
            // 1. Arc length filter to reject small noise
            double arcLen = Imgproc.arcLength(contour2f, false);
            if (arcLen < minArcLength) continue;
            
            // 2. Need at least 5 points to fit an ellipse
            if (contour2f.rows() >= 5) {
                RotatedRect rect = Imgproc.fitEllipse(contour2f);
                
                // 3. Area ratio filter (SAM CDA paper approach)
                double contourArea = Imgproc.contourArea(contour);
                double a_val = rect.size.width / 2.0;
                double b_val = rect.size.height / 2.0;
                double ellipseArea = Math.PI * a_val * b_val;
                
                if (ellipseArea == 0) continue;
                double ratio = contourArea / ellipseArea;
                if (ratio < 0.7 || ratio > 1.3) continue; // Not a crater
                
                // 4. Aspect ratio filter (min / max axis)
                double minAxis = Math.min(rect.size.width, rect.size.height);
                double maxAxis = Math.max(rect.size.width, rect.size.height);
                
                if (maxAxis == 0) continue;
                double aspectRatio = minAxis / maxAxis;
                
                if (aspectRatio >= minAspectRatio) {
                    // int[]{cx, cy, a, b, angle}
                    // a and b are semi-axes, so width/2 and height/2
                    int cx = (int) Math.round(rect.center.x);
                    int cy = (int) Math.round(rect.center.y);
                    int a = (int) Math.round(rect.size.width / 2.0);
                    int b = (int) Math.round(rect.size.height / 2.0);
                    int angle = (int) Math.round(rect.angle);
                    
                    result.add(new int[]{cx, cy, a, b, angle});
                }
            }
        }
        
        hierarchy.release();
        return result;
    }

    public double getMinArcLength() { return minArcLength; }
    public void setMinArcLength(double v) { this.minArcLength = v; }

    public double getMinAspectRatio() { return minAspectRatio; }
    public void setMinAspectRatio(double v) { this.minAspectRatio = v; }
}
