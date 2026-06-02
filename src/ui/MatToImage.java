package ui;

import javafx.scene.image.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

/**
 * Utility: converts OpenCV Mat → JavaFX WritableImage for display.
 */
public class MatToImage {

    public static Image toFXImage(Mat mat) {
        if (mat == null || mat.empty()) return null;

        Mat display = new Mat();

        // Ensure BGR (3-channel) for display
        if (mat.channels() == 1) {
            Imgproc.cvtColor(mat, display, Imgproc.COLOR_GRAY2BGR);
        } else if (mat.channels() == 4) {
            Imgproc.cvtColor(mat, display, Imgproc.COLOR_BGRA2BGR);
        } else {
            mat.copyTo(display);
        }

        int w = display.cols();
        int h = display.rows();
        byte[] data = new byte[w * h * 3];
        display.get(0, 0, data);
        display.release();

        WritableImage wi = new WritableImage(w, h);
        PixelWriter pw = wi.getPixelWriter();

        // OpenCV stores BGR; JavaFX wants ARGB → convert per pixel
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 3;
                int b = data[idx]     & 0xFF;
                int g = data[idx + 1] & 0xFF;
                int r = data[idx + 2] & 0xFF;
                pw.setArgb(x, y, (0xFF << 24) | (r << 16) | (g << 8) | b);
            }
        }

        return wi;
    }
}
