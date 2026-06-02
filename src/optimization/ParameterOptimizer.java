package optimization;

import pipeline.CraterPipeline;
import pipeline.PipelineResult;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class ParameterOptimizer {
    // Load OpenCV native library
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) throws Exception {
        System.out.println("====================================================");
        System.out.println("  LunarCraterDetector — Automated Optimizer  ");
        System.out.println("====================================================");
        
        File imagesDir = new File("resources/dataset/train/images");
        File labelsDir = new File("resources/dataset/train/labels");
        
        if (!imagesDir.exists() || !labelsDir.exists()) {
            System.err.println("Dataset not found! Please place Roboflow dataset inside:");
            System.err.println("  " + imagesDir.getAbsolutePath());
            System.err.println("  " + labelsDir.getAbsolutePath());
            return;
        }

        File[] imageFiles = imagesDir.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));
        if (imageFiles == null || imageFiles.length == 0) {
            System.err.println("No images found in dataset.");
            return;
        }
        
        // 80/20 train-val split
        List<File> allImages = new ArrayList<>(Arrays.asList(imageFiles));
        Collections.shuffle(allImages, new Random(42));
        int splitIdx = (int) (allImages.size() * 0.8);
        List<File> trainSet = allImages.subList(0, splitIdx);
        List<File> valSet = allImages.subList(splitIdx, allImages.size());
        
        System.out.println("Train set: " + trainSet.size() + " images");
        System.out.println("Validation set: " + valSet.size() + " images\n");

        CraterPipeline pipeline = new CraterPipeline();
        
        // Grid definitions for Grid Search
        double[] clipLimitArr = {1.5, 2.0, 3.0};
        double[] cannyT1Arr   = {30.0, 50.0};
        double[] cannyT2Arr   = {100.0, 150.0};
        double[] param2Arr    = {20.0, 30.0, 40.0};
        
        double bestF1 = -1;
        String bestJson = "";
        
        int totalCombos = clipLimitArr.length * cannyT1Arr.length * cannyT2Arr.length * param2Arr.length;
        int count = 0;

        System.out.println("Starting Grid Search (" + totalCombos + " combinations)...\n");

        for (double clip : clipLimitArr) {
            for (double t1 : cannyT1Arr) {
                for (double t2 : cannyT2Arr) {
                    for (double p2 : param2Arr) {
                        count++;
                        
                        pipeline.getPreprocessor().setClipLimit(clip);
                        pipeline.getEdgeDetector().setThreshold1(t1);
                        pipeline.getEdgeDetector().setThreshold2(t2);
                        pipeline.getLargeHoughDetector().setParam1(t2);
                        pipeline.getLargeHoughDetector().setParam2(p2);
                        pipeline.getSmallHoughDetector().setParam1(t2);
                        pipeline.getSmallHoughDetector().setParam2(p2);
                        
                        System.out.printf("[%d/%d] Evaluating clip=%.1f t1=%.0f t2=%.0f p2=%.0f ... ", 
                            count, totalCombos, clip, t1, t2, p2);
                            
                        Evaluator.Metrics totalMetrics = new Evaluator.Metrics();
                        
                        int imgCount = 0;
                        int totalImgs = trainSet.size();
                        for (File imgFile : trainSet) {
                            imgCount++;
                            System.out.print(String.format("\r[%d/%d] Evaluating clip=%.1f t1=%.0f t2=%.0f p2=%.0f ... Image %d/%d", 
                                count, totalCombos, clip, t1, t2, p2, imgCount, totalImgs));
                            System.out.flush();
                            String name = imgFile.getName();
                            String txtName = name.substring(0, name.lastIndexOf('.')) + ".txt";
                            File txtFile = new File(labelsDir, txtName);
                            
                            Mat img = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.IMREAD_COLOR);
                            if (img.empty()) continue;
                            
                            // (Resizing was removed because it shrinks craters below minRadius limits of HoughDetector/RegionFilter)
                            
                            List<int[]> truth = YoloParser.parse(txtFile, img.cols(), img.rows());
                            
                            PipelineResult res = pipeline.run(img);
                            List<int[]> detected = res.getCraters();
                            img.release();
                            
                            Evaluator.Metrics m = Evaluator.evaluate(detected, truth);
                            totalMetrics.tp += m.tp;
                            totalMetrics.fp += m.fp;
                            totalMetrics.fn += m.fn;
                            
                            // Prevent memory leak
                            if (res.getMat(PipelineResult.Stage.ORIGINAL) != null) res.getMat(PipelineResult.Stage.ORIGINAL).release();
                            if (res.getMat(PipelineResult.Stage.PREPROCESSED) != null) res.getMat(PipelineResult.Stage.PREPROCESSED).release();
                            if (res.getMat(PipelineResult.Stage.EDGES) != null) res.getMat(PipelineResult.Stage.EDGES).release();
                            if (res.getMat(PipelineResult.Stage.DETECTED) != null) res.getMat(PipelineResult.Stage.DETECTED).release();
                        }
                        
                        double p = totalMetrics.precision();
                        double r = totalMetrics.recall();
                        double f1 = totalMetrics.f1();
                        
                        System.out.printf(" Done! -> P: %.3f | R: %.3f | F1: %.3f%n", p, r, f1);
                            
                        if (f1 > bestF1) {
                            bestF1 = f1;
                            bestJson = String.format("{\n" +
                                    "  \"claheClipLimit\": %.2f,\n" +
                                    "  \"cannyT1\": %.2f,\n" +
                                    "  \"cannyT2\": %.2f,\n" +
                                    "  \"houghParam2\": %.2f\n" +
                                    "}", clip, t1, t2, p2);
                        }
                    }
                }
            }
        }
        
        System.out.println("\n====================================================");
        System.out.println("Grid Search Finished.");
        System.out.println("Best Train F1 Score: " + bestF1);
        System.out.println("Optimal Parameters:\n" + bestJson);
        
        File outFile = new File("resources/OptimalParams.json");
        outFile.getParentFile().mkdirs();
        try (FileWriter fw = new FileWriter(outFile)) {
            fw.write(bestJson);
        }
        System.out.println("Saved optimal parameters to: " + outFile.getAbsolutePath());
    }
}
