package ui;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import org.opencv.core.*;

import pipeline.CraterPipeline;
import pipeline.PipelineResult;

import java.io.File;
import java.nio.file.Files;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Main JavaFX window for Lunar Crater Detector.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────┐
 *   │  Title bar                                   │
 *   ├────────────┬─────────────────────────────────┤
 *   │  Controls  │  4-panel image display           │
 *   │  (left)    │  [Original] [Preprocessed]      │
 *   │            │  [Edges]    [Detected]           │
 *   ├────────────┴─────────────────────────────────┤
 *   │  Log / stats bar                             │
 *   └──────────────────────────────────────────────┘
 */
public class MainWindow extends Application {

    // Load OpenCV native lib once
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    private final CraterPipeline pipeline = new CraterPipeline();
    private PipelineResult lastResult;

    // Image views for the 4 panels
    private final ImageView ivOriginal     = makeImageView();
    private final ImageView ivPreprocessed = makeImageView();
    private final ImageView ivEdges        = makeImageView();
    private final ImageView ivDetected     = makeImageView();

    // Status / log
    private final TextArea  logArea        = new TextArea();
    private final Label     statusLabel    = new Label("Load an image to begin.");

    // Current loaded image path
    private String currentImagePath = null;
    
    // --- Parameter sliders (bound to pipeline) ---
    private Slider sClipLimit, sTileSize, sTopHatKernel, sMorphKernel, sMorphIters, sGaussKernel, sGaussSigma;
    private Slider sCannyT1, sCannyT2;
    private Slider sMorphCloseKernel, sMorphCloseIters;
    private Slider sLargeHoughMinDist, sLargeHoughParam1, sLargeHoughParam2, sLargeHoughMinR, sLargeHoughMaxR;
    private Slider sSmallHoughMinDist, sSmallHoughParam1, sSmallHoughParam2, sSmallHoughMinR, sSmallHoughMaxR;
    private Slider sCircularity, sMinArcLength, sQuadrantRatio;
    private ToggleButton tbtnAdvanced;

    private final ProgressBar progressBar = new ProgressBar(0);
    private final Label timerLabel = new Label("00:00.000");
    private javafx.animation.AnimationTimer timer;
    private long startTime;

    @Override
    public void start(Stage stage) {
        stage.setTitle("Lunar Crater Detector — COMP4687");

        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: #0d1117;");

        root.setTop(buildTitleBar());
        root.setLeft(buildControlPanel());
        root.setCenter(buildImageGrid());
        root.setBottom(buildLogBar());

        Scene scene = new Scene(root, 1400, 860);
        java.net.URL cssUrl = getClass().getResource("/style.css");
        if (cssUrl != null) {
            scene.getStylesheets().add(cssUrl.toExternalForm());
        }

        stage.setScene(scene);
        stage.setMinWidth(1000);
        stage.setMinHeight(700);
        stage.show();
        
        // Load default settings
        loadSettings();
    }

    // ------------------------------------------------------------------ UI builders

    private HBox buildTitleBar() {
        Label title = new Label("🌑  Lunar Crater Detector");
        title.setFont(Font.font("Monospaced", FontWeight.BOLD, 18));
        title.setTextFill(Color.web("#e6edf3"));

        Label sub = new Label("COMP4687 · Computer Vision · Işık University");
        sub.setFont(Font.font("Monospaced", 12));
        sub.setTextFill(Color.web("#8b949e"));

        Button btnLoad = styledButton("📂 Load Image", "#238636");
        btnLoad.setOnAction(e -> loadImage());

        Button btnRun = styledButton("▶ Run Detection", "#1f6feb");
        btnRun.setOnAction(e -> runPipeline());

        Button btnClear = styledButton("✕ Clear", "#30363d");
        btnClear.setOnAction(e -> clearAll());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox bar = new HBox(14, title, sub, spacer, btnLoad, btnRun, btnClear);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.setPadding(new Insets(12, 16, 12, 16));
        bar.setStyle("-fx-background-color: #161b22; -fx-border-color: #30363d; -fx-border-width: 0 0 1 0;");
        return bar;
    }

    private ScrollPane buildControlPanel() {
        VBox panel = new VBox(8);
        panel.setPadding(new Insets(12));
        panel.setPrefWidth(240);
        panel.setStyle("-fx-background-color: #161b22;");

        tbtnAdvanced = new ToggleButton("⚙ Gelişmiş Ayarlar (Advanced Mode)");
        tbtnAdvanced.setMaxWidth(Double.MAX_VALUE);
        tbtnAdvanced.setStyle("-fx-base: #21262d; -fx-text-fill: #c9d1d9; -fx-font-family: monospace; -fx-cursor: hand;");

        VBox advancedVBox = new VBox(8);
        advancedVBox.setVisible(false);
        advancedVBox.setManaged(false);

        tbtnAdvanced.selectedProperty().addListener((o, ov, nv) -> {
            advancedVBox.setVisible(nv);
            advancedVBox.setManaged(nv);
        });

        panel.getChildren().add(tbtnAdvanced);

        sMorphCloseKernel = addSliderInt(advancedVBox, "Closing Kernel", 1, 31, 5,
            v -> pipeline.getEdgeDetector().setMorphCloseKernel(v));
        sMorphCloseIters = addSliderInt(advancedVBox, "Closing Iterations", 0, 10, 1,
            v -> pipeline.getEdgeDetector().setMorphCloseIters(v));
            
        advancedVBox.getChildren().add(new Separator());

        // --- Preprocessing ---
        advancedVBox.getChildren().add(sectionLabel("Preprocessing"));
        sClipLimit   = addSlider(advancedVBox, "CLAHE Clip Limit",  0.5, 16.0, 2.0,
            v -> pipeline.getPreprocessor().setClipLimit(v));
        sTileSize    = addSliderInt(advancedVBox, "CLAHE Tile Size",    4, 16,  8,
            v -> pipeline.getPreprocessor().setTileSize(v));
        sTopHatKernel = addSliderInt(advancedVBox, "Top-Hat Kernel",    3, 151, 15,
            v -> pipeline.getPreprocessor().setTopHatKernel(v));
        sMorphKernel = addSliderInt(advancedVBox, "Morph Kernel",       1, 15,  3,
            v -> pipeline.getPreprocessor().setMorphKernel(v));
        sMorphIters  = addSliderInt(advancedVBox, "Morph Iterations",   1, 10,  1,
            v -> pipeline.getPreprocessor().setMorphIterations(v));
        sGaussKernel = addSliderInt(advancedVBox, "Gauss Kernel",       3, 51,  5,
            v -> pipeline.getPreprocessor().setGaussKernel(v));
        sGaussSigma  = addSlider(advancedVBox, "Gauss Sigma",       0.5, 15.0, 1.5,
            v -> pipeline.getPreprocessor().setGaussSigma(v));
        panel.getChildren().add(new Separator());

        // --- Canny ---
        advancedVBox.getChildren().add(sectionLabel("Canny Parameters"));
        sCannyT1 = addSlider(advancedVBox, "Threshold 1",  10, 200,  50,
            v -> pipeline.getEdgeDetector().setThreshold1(v));
        sCannyT2 = addSlider(advancedVBox, "Threshold 2",  50, 400, 150,
            v -> pipeline.getEdgeDetector().setThreshold2(v));
        panel.getChildren().add(new Separator());

        // --- Hough (Large) ---
        advancedVBox.getChildren().add(sectionLabel("Hough: Large Craters"));
        sLargeHoughMinDist = addSlider(advancedVBox, "Min Distance",   5, 400, 100,
            v -> pipeline.getLargeHoughDetector().setMinDist(v));
        sLargeHoughParam1  = addSlider(advancedVBox, "Canny Param1", 10, 200, 100,
            v -> pipeline.getLargeHoughDetector().setParam1(v));
        sLargeHoughParam2  = addSlider(advancedVBox, "Accumulator Th.", 5, 100, 60,
            v -> pipeline.getLargeHoughDetector().setParam2(v));
        sLargeHoughMinR    = addSliderInt(advancedVBox, "Min Radius",  3, 300, 80,
            v -> pipeline.getLargeHoughDetector().setMinRadius(v));
        sLargeHoughMaxR    = addSliderInt(advancedVBox, "Max Radius",  20, 800, 400,
            v -> pipeline.getLargeHoughDetector().setMaxRadius(v));
        panel.getChildren().add(new Separator());

        // --- Hough (Small) ---
        advancedVBox.getChildren().add(sectionLabel("Hough: Small Craters"));
        sSmallHoughMinDist = addSlider(advancedVBox, "Min Distance",   5, 400, 30,
            v -> pipeline.getSmallHoughDetector().setMinDist(v));
        sSmallHoughParam1  = addSlider(advancedVBox, "Canny Param1", 10, 200, 80,
            v -> pipeline.getSmallHoughDetector().setParam1(v));
        sSmallHoughParam2  = addSlider(advancedVBox, "Accumulator Th.", 5, 100, 35,
            v -> pipeline.getSmallHoughDetector().setParam2(v));
        sSmallHoughMinR    = addSliderInt(advancedVBox, "Min Radius",  3, 300, 15,
            v -> pipeline.getSmallHoughDetector().setMinRadius(v));
        sSmallHoughMaxR    = addSliderInt(advancedVBox, "Max Radius",  20, 800, 80,
            v -> pipeline.getSmallHoughDetector().setMaxRadius(v));
        advancedVBox.getChildren().add(new Separator());

        // --- Ellipse ---
        advancedVBox.getChildren().add(sectionLabel("Ellipse Detector"));
        sMinArcLength = addSlider(advancedVBox, "Min Arc Length", 10.0, 200.0, 30.0,
            v -> pipeline.getEllipseDetector().setMinArcLength(v));
        advancedVBox.getChildren().add(new Separator());

        // --- Region filter ---
        advancedVBox.getChildren().add(sectionLabel("Region Filter"));
        sCircularity  = addSlider(advancedVBox, "Min Circularity", 0.0, 1.0, 0.15,
            v -> pipeline.getRegionFilter().setMinCircularity(v));
        sQuadrantRatio = addSlider(advancedVBox, "Max Quadrant Ratio", 1.5, 10.0, 3.0,
            v -> pipeline.getRegionFilter().setMaxQuadrantRatio(v));

        // auto threshold button
        Button btnAutoThresh = styledButton("🪄 Auto Threshold (Otsu)", "#0277bd");
        btnAutoThresh.setMaxWidth(Double.MAX_VALUE);
        btnAutoThresh.setOnAction(e -> applyAutoThreshold());

        // reset button
        Button btnReset = styledButton("↺ Reset Defaults", "#30363d");
        btnReset.setMaxWidth(Double.MAX_VALUE);
        btnReset.setOnAction(e -> resetDefaults());

        // save settings button
        Button btnSave = styledButton("💾 Save Settings", "#8957e5");
        btnSave.setMaxWidth(Double.MAX_VALUE);
        btnSave.setOnAction(e -> saveSettings());

        advancedVBox.getChildren().addAll(new Separator(), btnAutoThresh, btnReset, btnSave);
        panel.getChildren().add(advancedVBox);

        ScrollPane sp = new ScrollPane(panel);
        sp.setFitToWidth(true);
        sp.setStyle("-fx-background-color: #161b22; -fx-border-color: #30363d; -fx-border-width: 0 1 0 0;");
        return sp;
    }

    private GridPane buildImageGrid() {
        GridPane grid = new GridPane();
        grid.setHgap(6); grid.setVgap(6);
        grid.setPadding(new Insets(8));
        grid.setStyle("-fx-background-color: #0d1117;");

        ColumnConstraints cc1 = new ColumnConstraints();
        cc1.setPercentWidth(50); cc1.setHgrow(Priority.ALWAYS);
        ColumnConstraints cc2 = new ColumnConstraints();
        cc2.setPercentWidth(50); cc2.setHgrow(Priority.ALWAYS);
        grid.getColumnConstraints().addAll(cc1, cc2);

        RowConstraints rc1 = new RowConstraints();
        rc1.setPercentHeight(50); rc1.setVgrow(Priority.ALWAYS);
        RowConstraints rc2 = new RowConstraints();
        rc2.setPercentHeight(50); rc2.setVgrow(Priority.ALWAYS);
        grid.getRowConstraints().addAll(rc1, rc2);

        grid.add(imagePanel("Original",     ivOriginal),     0, 0);
        grid.add(imagePanel("Preprocessed", ivPreprocessed), 1, 0);
        grid.add(imagePanel("Edge Detection",                  ivEdges),        0, 1);
        grid.add(imagePanel("Detected Craters",                ivDetected),     1, 1);

        ivDetected.setOnMouseClicked(e -> handleCraterClick(e));

        return grid;
    }

    private VBox imagePanel(String title, ImageView iv) {
        Label lbl = new Label(title);
        lbl.setFont(Font.font("Monospaced", 11));
        lbl.setTextFill(Color.web("#8b949e"));
        lbl.setPadding(new Insets(4, 6, 4, 6));

        StackPane pane = new StackPane(iv);
        pane.setStyle("-fx-background-color: #010409;");
        pane.setAlignment(Pos.CENTER);
        VBox.setVgrow(pane, Priority.ALWAYS);

        VBox box = new VBox(0, lbl, pane);
        box.setStyle("-fx-background-color: #161b22; -fx-border-color: #30363d; -fx-border-width: 1;");
        return box;
    }

    private HBox buildLogBar() {
        logArea.setEditable(false);
        logArea.setPrefHeight(90);
        logArea.setStyle("-fx-control-inner-background: #0d1117; -fx-text-fill: #8b949e; -fx-font-family: monospace; -fx-font-size: 11;");

        statusLabel.setTextFill(Color.web("#3fb950"));
        statusLabel.setFont(Font.font("Monospaced", 12));
        statusLabel.setPadding(new Insets(4, 8, 4, 8));

        progressBar.setPrefWidth(200);
        progressBar.setVisible(false);
        
        timerLabel.setTextFill(Color.web("#e3b341"));
        timerLabel.setFont(Font.font("Monospaced", 12));
        timerLabel.setVisible(false);

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        HBox topBar = new HBox(10, statusLabel, spacer, progressBar, timerLabel);
        topBar.setAlignment(Pos.CENTER_LEFT);

        VBox box = new VBox(2, topBar, logArea);
        box.setPadding(new Insets(6, 8, 6, 8));
        box.setStyle("-fx-background-color: #161b22; -fx-border-color: #30363d; -fx-border-width: 1 0 0 0;");
        return new HBox(box);
    }

    // ------------------------------------------------------------------ Actions

    private void loadImage() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Select Lunar Image");
        fc.getExtensionFilters().addAll(
            new FileChooser.ExtensionFilter("Image Files", "*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp")
        );
        File f = fc.showOpenDialog(null);
        if (f == null) return;

        currentImagePath = f.getAbsolutePath();
        statusLabel.setText("Loaded: " + f.getName());
        logArea.appendText("Image loaded: " + currentImagePath + "\n");

        // Show original immediately
        Mat mat = org.opencv.imgcodecs.Imgcodecs.imread(currentImagePath);
        if (!mat.empty()) setImageView(ivOriginal, mat);
    }

    private void applyAutoThreshold() {
        if (currentImagePath == null) {
            statusLabel.setText("⚠ Load an image first!");
            statusLabel.setTextFill(Color.web("#e3b341"));
            return;
        }

        // Load image in grayscale
        Mat gray = org.opencv.imgcodecs.Imgcodecs.imread(currentImagePath, org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE);
        if (gray.empty()) return;

        // Apply a light blur to reduce noise before Otsu calculation
        Mat blurred = new Mat();
        org.opencv.imgproc.Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 1.5);

        // Calculate Otsu's threshold
        Mat temp = new Mat();
        double otsuThresh = org.opencv.imgproc.Imgproc.threshold(blurred, temp, 0, 255,
            org.opencv.imgproc.Imgproc.THRESH_BINARY | org.opencv.imgproc.Imgproc.THRESH_OTSU);

        // Optimal parameters based on Otsu
        double t2 = otsuThresh;
        double t1 = otsuThresh * 0.5;

        // Ensure within valid bounds
        t2 = Math.min(400, Math.max(20, t2));
        t1 = Math.min(200, Math.max(5, t1));

        // Update UI Sliders
        sCannyT1.setValue(t1);
        sCannyT2.setValue(t2);
        sLargeHoughParam1.setValue(t2);
        sSmallHoughParam1.setValue(t2 * 0.8); // Slightly lower for small craters

        logArea.appendText(String.format("Auto Threshold (Otsu) calculated: %.1f\n", otsuThresh));
        logArea.appendText(String.format(" -> Canny T1: %.1f, Canny T2: %.1f, L-Hough P1: %.1f, S-Hough P1: %.1f\n", t1, t2, t2, t2 * 0.8));
        statusLabel.setText("Auto threshold complete. Run detection.");
        statusLabel.setTextFill(Color.web("#3fb950"));

        gray.release();
        blurred.release();
        temp.release();
    }

    private void runPipeline() {
        if (currentImagePath == null) {
            statusLabel.setText("⚠ No image loaded.");
            return;
        }
        
        if (tbtnAdvanced != null && !tbtnAdvanced.isSelected()) {
            applyAutoThreshold();
        }
        
        statusLabel.setText("⏳ Running pipeline...");
        statusLabel.setTextFill(Color.web("#e3b341"));
        
        progressBar.setProgress(0);
        progressBar.setVisible(true);
        timerLabel.setText("00:00.000");
        timerLabel.setVisible(true);

        startTime = System.currentTimeMillis();
        timer = new javafx.animation.AnimationTimer() {
            @Override
            public void handle(long now) {
                long elapsed = System.currentTimeMillis() - startTime;
                long seconds = elapsed / 1000;
                long millis = elapsed % 1000;
                timerLabel.setText(String.format("%02d:%02d.%03d", seconds / 60, seconds % 60, millis));
            }
        };
        timer.start();

        Task<PipelineResult> task = new Task<>() {
            @Override protected PipelineResult call() {
                return pipeline.run(currentImagePath, val -> {
                    Platform.runLater(() -> progressBar.setProgress(val));
                });
            }
        };

        task.setOnSucceeded(e -> {
            if (timer != null) timer.stop();
            progressBar.setVisible(false);
            timerLabel.setVisible(false);
            PipelineResult r = task.getValue();
            lastResult = r;
            setImageView(ivOriginal,     r.getMat(PipelineResult.Stage.ORIGINAL));
            setImageView(ivPreprocessed, r.getMat(PipelineResult.Stage.PREPROCESSED));
            setImageView(ivEdges,        r.getMat(PipelineResult.Stage.EDGES));
            setImageView(ivDetected,     r.getMat(PipelineResult.Stage.DETECTED));

            logArea.appendText(r.getLog());
            logArea.appendText("─────────────────────────────\n");
            statusLabel.setText("✓ " + r.getCraterCount() + " craters detected  |  " +
                r.getProcessingTimeMs() + " ms");
            statusLabel.setTextFill(Color.web("#3fb950"));
        });

        task.setOnFailed(e -> {
            if (timer != null) timer.stop();
            progressBar.setVisible(false);
            timerLabel.setVisible(false);
            statusLabel.setText("✗ Pipeline error: " + task.getException().getMessage());
            statusLabel.setTextFill(Color.web("#f85149"));
            task.getException().printStackTrace();
        });

        new Thread(task, "pipeline-thread").start();
    }

    private void clearAll() {
        ivOriginal.setImage(null);
        ivPreprocessed.setImage(null);
        ivEdges.setImage(null);
        ivDetected.setImage(null);
        logArea.clear();
        statusLabel.setText("Cleared.");
        statusLabel.setTextFill(Color.web("#8b949e"));
        currentImagePath = null;
    }

    private void resetDefaults() {
        sClipLimit.setValue(2.0); sTileSize.setValue(8);
        sTopHatKernel.setValue(15);
        sMorphKernel.setValue(3); sMorphIters.setValue(1);
        sGaussKernel.setValue(5); sGaussSigma.setValue(1.5);
        sCannyT1.setValue(50);    sCannyT2.setValue(150);
        sMorphCloseKernel.setValue(5); sMorphCloseIters.setValue(1);
        
        sLargeHoughMinDist.setValue(100); sLargeHoughParam1.setValue(100); sLargeHoughParam2.setValue(60);
        sLargeHoughMinR.setValue(80);     sLargeHoughMaxR.setValue(400);

        sSmallHoughMinDist.setValue(30);  sSmallHoughParam1.setValue(80);  sSmallHoughParam2.setValue(35);
        sSmallHoughMinR.setValue(15);     sSmallHoughMaxR.setValue(80);
        sMinArcLength.setValue(30.0);
        sCircularity.setValue(0.15);
        sQuadrantRatio.setValue(3.0);
    }

    private void saveSettings() {
        try {
            String file = "resources/Params.json";
            String json = String.format("{\n" +
                "  \"claheClipLimit\": %.2f,\n  \"claheTileSize\": %d,\n  \"topHatKernel\": %d,\n  \"morphKernel\": %d,\n  \"morphIters\": %d,\n  \"gaussKernel\": %d,\n  \"gaussSigma\": %.2f,\n" +
                "  \"cannyT1\": %.2f,\n  \"cannyT2\": %.2f,\n" +
                "  \"morphCloseKernel\": %d,\n  \"morphCloseIters\": %d,\n" +
                "  \"largeHoughMinDist\": %.2f,\n  \"largeHoughParam1\": %.2f,\n  \"largeHoughParam2\": %.2f,\n  \"largeHoughMinR\": %d,\n  \"largeHoughMaxR\": %d,\n" +
                "  \"smallHoughMinDist\": %.2f,\n  \"smallHoughParam1\": %.2f,\n  \"smallHoughParam2\": %.2f,\n  \"smallHoughMinR\": %d,\n  \"smallHoughMaxR\": %d,\n" +
                "  \"minArcLength\": %.2f,\n  \"circularity\": %.2f,\n  \"quadrantRatio\": %.2f\n}", 
                sClipLimit.getValue(), (int)sTileSize.getValue(), (int)sTopHatKernel.getValue(), (int)sMorphKernel.getValue(), (int)sMorphIters.getValue(), (int)sGaussKernel.getValue(), sGaussSigma.getValue(),
                sCannyT1.getValue(), sCannyT2.getValue(), 
                (int)sMorphCloseKernel.getValue(), (int)sMorphCloseIters.getValue(),
                sLargeHoughMinDist.getValue(), sLargeHoughParam1.getValue(), sLargeHoughParam2.getValue(), (int)sLargeHoughMinR.getValue(), (int)sLargeHoughMaxR.getValue(),
                sSmallHoughMinDist.getValue(), sSmallHoughParam1.getValue(), sSmallHoughParam2.getValue(), (int)sSmallHoughMinR.getValue(), (int)sSmallHoughMaxR.getValue(),
                sMinArcLength.getValue(), sCircularity.getValue(), sQuadrantRatio.getValue()
            );
            Files.writeString(new File(file).toPath(), json);
            statusLabel.setText("✓ Saved settings.");
            statusLabel.setTextFill(Color.web("#3fb950"));
            logArea.appendText("Saved parameters to " + file + "\n");
        } catch(Exception e) {
            e.printStackTrace();
            statusLabel.setText("✗ Failed to save settings.");
            statusLabel.setTextFill(Color.web("#f85149"));
        }
    }

    private void loadSettings() {
        try {
            String file = "resources/Params.json";
            File jsonFile = new File(file);
            if (!jsonFile.exists()) return;
            
            String content = Files.readString(jsonFile.toPath());
            
            doubleVal(content, "claheClipLimit", sClipLimit::setValue);
            intVal(content, "claheTileSize", v -> sTileSize.setValue(v));
            intVal(content, "topHatKernel", v -> sTopHatKernel.setValue(v));
            intVal(content, "morphKernel", v -> sMorphKernel.setValue(v));
            intVal(content, "morphIters", v -> sMorphIters.setValue(v));
            intVal(content, "gaussKernel", v -> sGaussKernel.setValue(v));
            doubleVal(content, "gaussSigma", sGaussSigma::setValue);
            doubleVal(content, "cannyT1", sCannyT1::setValue);
            doubleVal(content, "cannyT2", sCannyT2::setValue);
            intVal(content, "morphCloseKernel", v -> sMorphCloseKernel.setValue(v));
            intVal(content, "morphCloseIters", v -> sMorphCloseIters.setValue(v));
            
            // New large params
            doubleVal(content, "largeHoughMinDist", sLargeHoughMinDist::setValue);
            doubleVal(content, "largeHoughParam1", sLargeHoughParam1::setValue);
            doubleVal(content, "largeHoughParam2", sLargeHoughParam2::setValue);
            intVal(content, "largeHoughMinR", v -> sLargeHoughMinR.setValue(v));
            intVal(content, "largeHoughMaxR", v -> sLargeHoughMaxR.setValue(v));

            // New small params
            doubleVal(content, "smallHoughMinDist", sSmallHoughMinDist::setValue);
            doubleVal(content, "smallHoughParam1", sSmallHoughParam1::setValue);
            doubleVal(content, "smallHoughParam2", sSmallHoughParam2::setValue);
            intVal(content, "smallHoughMinR", v -> sSmallHoughMinR.setValue(v));
            intVal(content, "smallHoughMaxR", v -> sSmallHoughMaxR.setValue(v));

            doubleVal(content, "minArcLength", sMinArcLength::setValue);
            doubleVal(content, "circularity", sCircularity::setValue);
            doubleVal(content, "quadrantRatio", sQuadrantRatio::setValue);

            statusLabel.setText("✓ Loaded settings.");
            statusLabel.setTextFill(Color.web("#3fb950"));
            logArea.appendText("Loaded parameters from " + file + "\n");
        } catch (Exception ex) {
            statusLabel.setText("✗ Error loading params: " + ex.getMessage());
            statusLabel.setTextFill(Color.web("#f85149"));
            ex.printStackTrace();
        }
    }

    private void doubleVal(String content, String key, java.util.function.DoubleConsumer setter) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*([0-9.]+)").matcher(content);
        if (m.find()) setter.accept(Double.parseDouble(m.group(1)));
    }
    
    private void intVal(String content, String key, java.util.function.IntConsumer setter) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*([0-9.]+)").matcher(content);
        if (m.find()) setter.accept((int)Double.parseDouble(m.group(1)));
    }

    // ------------------------------------------------------------------ Helpers

    private void setImageView(ImageView iv, Mat mat) {
        if (mat == null || mat.empty()) return;
        Image img = MatToImage.toFXImage(mat);
        Platform.runLater(() -> {
            iv.setImage(img);
            iv.setFitWidth(((StackPane) iv.getParent()).getWidth());
            iv.setFitHeight(((StackPane) iv.getParent()).getHeight());
        });
    }

    private static ImageView makeImageView() {
        ImageView iv = new ImageView();
        iv.setPreserveRatio(true);
        iv.setSmooth(true);
        return iv;
    }

    private Label sectionLabel(String text) {
        Label l = new Label(text.toUpperCase());
        l.setFont(Font.font("Monospaced", FontWeight.BOLD, 10));
        l.setTextFill(Color.web("#3fb950"));
        l.setPadding(new Insets(8, 0, 2, 0));
        return l;
    }

    private Button styledButton(String text, String bg) {
        Button b = new Button(text);
        b.setStyle("-fx-background-color: " + bg + "; -fx-text-fill: #e6edf3; " +
            "-fx-font-family: monospace; -fx-cursor: hand; -fx-border-radius: 4; -fx-background-radius: 4;");
        return b;
    }

    private void handleCraterClick(javafx.scene.input.MouseEvent e) {
        if (lastResult == null || lastResult.getMat(PipelineResult.Stage.ORIGINAL) == null) return;
        
        // Calculate image coordinates based on ImageView scale and letterboxing
        double viewW = ivDetected.getBoundsInLocal().getWidth();
        double viewH = ivDetected.getBoundsInLocal().getHeight();
        
        Mat orig = lastResult.getMat(PipelineResult.Stage.ORIGINAL);
        double imgW = orig.width();
        double imgH = orig.height();
        
        double scale = Math.min(viewW / imgW, viewH / imgH);
        
        double actualW = imgW * scale;
        double actualH = imgH * scale;
        
        double offsetX = (viewW - actualW) / 2;
        double offsetY = (viewH - actualH) / 2;
        
        double mouseX = e.getX();
        double mouseY = e.getY();
        
        if (mouseX < offsetX || mouseX > offsetX + actualW ||
            mouseY < offsetY || mouseY > offsetY + actualH) {
            return;
        }
        
        double imgX = (mouseX - offsetX) / scale;
        double imgY = (mouseY - offsetY) / scale;
        
        // Find nearest crater
        int[] nearest = null;
        double minDist = Double.MAX_VALUE;
        for (int[] c : lastResult.getCraters()) {
            double dx = c[0] - imgX;
            double dy = c[1] - imgY;
            double dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < c[2] && dist < minDist) {
                minDist = dist;
                nearest = c;
            }
        }
        
        if (nearest != null) {
            showCraterZoom(nearest);
        }
    }

    private void showCraterZoom(int[] crater) {
        Mat orig = lastResult.getMat(PipelineResult.Stage.ORIGINAL);
        int cx = crater[0];
        int cy = crater[1];
        int r  = crater[2];
        
        int boxSize = r * 4;
        int x1 = Math.max(0, cx - boxSize / 2);
        int y1 = Math.max(0, cy - boxSize / 2);
        int x2 = Math.min(orig.cols(), cx + boxSize / 2);
        int y2 = Math.min(orig.rows(), cy + boxSize / 2);
        
        Rect roi = new Rect(x1, y1, x2 - x1, y2 - y1);
        Mat cropped = new Mat(orig, roi);
        
        Mat resized = new Mat();
        org.opencv.imgproc.Imgproc.resize(cropped, resized, new Size(256, 256), 0, 0, org.opencv.imgproc.Imgproc.INTER_CUBIC);
        
        // Draw crater circle on the cropped image
        org.opencv.imgproc.Imgproc.circle(resized, new Point(256/2.0, 256/2.0), (int)((r * 256.0) / boxSize), new Scalar(0, 255, 0), 2);
        
        Image img = MatToImage.toFXImage(resized);
        ImageView iv = new ImageView(img);
        iv.setFitWidth(256); iv.setFitHeight(256);
        
        VBox root = new VBox(10);
        root.setPadding(new Insets(10));
        root.setStyle("-fx-background-color: #0d1117;");
        root.setAlignment(Pos.CENTER);
        
        Label info = new Label(String.format("Center: (%d, %d)\nRadius: %d", cx, cy, r));
        info.setFont(Font.font("Monospaced", 14));
        info.setTextFill(Color.web("#e6edf3"));
        
        root.getChildren().addAll(iv, info);
        
        Stage stage = new Stage();
        stage.setTitle("Crater Details");
        stage.setScene(new Scene(root));
        stage.show();
        
        resized.release();
    }

    // Generic slider factory (double)
    private Slider addSlider(VBox parent, String label, double min, double max, double def,
                              java.util.function.DoubleConsumer onChange) {
        Label lbl = new Label(label + ": " + String.format("%.2f", def));
        lbl.setFont(Font.font("Monospaced", 10));
        lbl.setTextFill(Color.web("#8b949e"));

        Slider s = new Slider(min, max, def);
        s.setShowTickLabels(false);
        s.setStyle("-fx-control-inner-background: #21262d;");

        s.valueProperty().addListener((o, ov, nv) -> {
            lbl.setText(label + ": " + String.format("%.2f", nv.doubleValue()));
            onChange.accept(nv.doubleValue());
        });

        parent.getChildren().addAll(lbl, s);
        return s;
    }

    // Integer slider factory
    private Slider addSliderInt(VBox parent, String label, int min, int max, int def,
                                 java.util.function.IntConsumer onChange) {
        Label lbl = new Label(label + ": " + def);
        lbl.setFont(Font.font("Monospaced", 10));
        lbl.setTextFill(Color.web("#8b949e"));

        Slider s = new Slider(min, max, def);
        s.setMajorTickUnit(1); s.setSnapToTicks(true);
        s.setStyle("-fx-control-inner-background: #21262d;");

        s.valueProperty().addListener((o, ov, nv) -> {
            int v = nv.intValue();
            lbl.setText(label + ": " + v);
            onChange.accept(v);
        });

        parent.getChildren().addAll(lbl, s);
        return s;
    }
}
