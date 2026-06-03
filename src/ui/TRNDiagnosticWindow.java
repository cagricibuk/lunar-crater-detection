package ui;

import algorithms.TemplateMatcher;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;
import org.opencv.core.Mat;
import pipeline.TRNDebugContext;
import java.awt.image.BufferedImage;

/**
 * TRN Diagnostic & GNC Telemetry Panel (Soft Edge ZNCC Mode).
 *
 * 3-column engineering dashboard that visualises the heavy Gaussian blur
 * "gravity field" effect, and reports ZNCC correlation metrics.
 */
public class TRNDiagnosticWindow extends Stage {

    private final TRNDebugContext ctx;

    public TRNDiagnosticWindow(TRNDebugContext ctx) {
        this.ctx = ctx;
        setTitle("AIDA — TRN Diagnostik (Soft Edge ZNCC Telemetri Paneli)");

        HBox root = new HBox(16);
        root.setPadding(new Insets(16));
        root.setStyle("-fx-background-color: #0d1117;");

        // ── Column 1: Input Views ──
        VBox col1 = createColumn("📡  Giriş Verileri");
        col1.getChildren().addAll(
            sectionTitle("ORİJİNAL HARİTA (50 km)"),
            makeBufferedImagePane(ctx.mapImage, true),
            sectionTitle("ORİJİNAL ŞABLON (5 km)"),
            makeBufferedImagePane(ctx.templateImage, false)
        );

        // ── Column 2: Intermediate Views ──
        VBox col2 = createColumn("🔬  Soft Edge (Gaussian) Dünyası");
        col2.getChildren().addAll(
            sectionTitle("HARİTA — 15x15 GAUSSIAN BULUTU"),
            makeEdgePane(ctx.mapSoftEdges),
            sectionTitle("ŞABLON — 15x15 GAUSSIAN BULUTU"),
            makeEdgePane(ctx.templateSoftEdges)
        );

        // ── Column 3: GNC Telemetry ──
        VBox col3 = createColumn("📊  GNC Telemetri (ZNCC)");
        col3.getChildren().add(buildTelemetry());

        // sizing
        col1.setPrefWidth(420);
        col2.setPrefWidth(420);
        col3.setPrefWidth(360);
        HBox.setHgrow(col1, Priority.ALWAYS);
        HBox.setHgrow(col2, Priority.ALWAYS);

        root.getChildren().addAll(col1, col2, col3);
        setScene(new Scene(root, 1350, 820));
    }

    // ────────────────────────── Layout helpers

    private VBox createColumn(String title) {
        VBox v = new VBox(8);
        v.setStyle("-fx-background-color: #161b22; -fx-border-color: #30363d; "
                 + "-fx-border-width: 1; -fx-border-radius: 10; -fx-background-radius: 10;");
        v.setPadding(new Insets(14));

        Label lbl = new Label(title);
        lbl.setFont(Font.font("System", FontWeight.BOLD, 16));
        lbl.setTextFill(Color.web("#c9d1d9"));
        v.getChildren().add(lbl);
        return v;
    }

    private Label sectionTitle(String text) {
        Label l = new Label(text);
        l.setFont(Font.font("Monospaced", FontWeight.BOLD, 11));
        l.setTextFill(Color.web("#58a6ff"));
        l.setPadding(new Insets(8, 0, 2, 0));
        return l;
    }

    // ── Image viewers ──

    private ScrollPane makeEdgePane(Mat mat) {
        if (mat == null || mat.empty()) return placeholder("Soft Edge verisi yok");
        Image img = MatToImage.toFXImage(mat);
        ImageView iv = new ImageView(img);
        iv.setPreserveRatio(true);
        iv.setFitWidth(390);
        return wrapScroll(iv);
    }

    private ScrollPane makeBufferedImagePane(BufferedImage bi, boolean drawBox) {
        if (bi == null) return placeholder("Görüntü yok");
        
        // Convert BufferedImage to JavaFX Image
        javafx.scene.image.WritableImage img = new javafx.scene.image.WritableImage(bi.getWidth(), bi.getHeight());
        javafx.embed.swing.SwingFXUtils.toFXImage(bi, img);

        Canvas canvas = new Canvas(img.getWidth(), img.getHeight());
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.drawImage(img, 0, 0);

        if (drawBox && ctx.finalMatch != null) {
            Color boxColor = voteColor(ctx.finalMatch);
            gc.setStroke(boxColor);
            gc.setLineWidth(4);
            gc.strokeRect(ctx.finalMatch.x, ctx.finalMatch.y,
                          ctx.finalMatch.width, ctx.finalMatch.height);

            // Label above box
            gc.setFill(boxColor);
            gc.setFont(javafx.scene.text.Font.font("Monospaced", FontWeight.BOLD, 18));
            String tag = String.format("Ölçek: %.3f | Skor: %.2f", ctx.finalMatch.optimalScale, ctx.finalMatch.score);
            gc.fillText(tag, ctx.finalMatch.x, Math.max(20, ctx.finalMatch.y - 8));
        }

        // Scale to column width visually
        javafx.scene.Group grp = new javafx.scene.Group(canvas);
        double scale = 390.0 / img.getWidth();
        if (scale < 1.0) { canvas.setScaleX(scale); canvas.setScaleY(scale); }

        return wrapScroll(grp);
    }

    private ScrollPane wrapScroll(javafx.scene.Node node) {
        ScrollPane sp = new ScrollPane(node);
        sp.setStyle("-fx-background-color: transparent; -fx-background: #0d1117;");
        sp.setPrefHeight(300);
        sp.setPannable(true);
        return sp;
    }

    private ScrollPane placeholder(String msg) {
        Label l = new Label(msg);
        l.setTextFill(Color.web("#8b949e"));
        l.setFont(Font.font("Monospaced", 12));
        ScrollPane sp = new ScrollPane(l);
        sp.setPrefHeight(100);
        sp.setStyle("-fx-background-color: transparent; -fx-background: #0d1117;");
        return sp;
    }

    // ────────────────────────── Telemetry panel

    private VBox buildTelemetry() {
        VBox panel = new VBox(10);
        panel.setPadding(new Insets(8));

        // ── Status banner ──
        TemplateMatcher.MatchResult m = ctx.finalMatch;
        String statusText;
        String color;
        if (m != null) {
            if (m.score >= 0.70)      { color = "#3fb950"; statusText = "HEDEF YAKALANDI"; }
            else if (m.score >= 0.50) { color = "#e3b341"; statusText = "OLASI EŞLEŞME"; }
            else                      { color = "#f85149"; statusText = "DÜŞÜK GÜVEN"; }
        } else {
            color = "#f85149"; statusText = "EŞLEŞME BULUNAMADI";
        }

        Label banner = new Label("═══  " + statusText + "  ═══");
        banner.setFont(Font.font("System", FontWeight.BOLD, 20));
        banner.setTextFill(Color.web(color));
        panel.getChildren().add(banner);

        panel.getChildren().add(separator());

        // ── Search section ──
        panel.getChildren().addAll(
            sectionTitle("ZNCC ARAMA METRİKLERİ"),
            metric("Hedeflenen Ölçek", String.format("%.3f", ctx.expectedScale)),
            metric("Taranan Pencere (Komb.)", String.format("%,d", ctx.totalZNCCWindowsScanned)),
            metric("Hesaplama Süresi", ctx.executionTimeMs + " ms")
        );

        // ── Result section ──
        if (m != null) {
            panel.getChildren().add(separator());
            panel.getChildren().addAll(
                sectionTitle("KAZANAN ZNCC EŞLEŞMESİ"),
                metric("Hesaplanan Ölçek", String.format("%.3f", m.optimalScale)),
                metric("ZNCC Özgüven Skoru", String.format("%.4f", m.score)),
                metric("Sınırlayıcı Kutu", String.format("%dx%d @ (%d,%d)", m.width, m.height, m.x, m.y))
            );
        }

        return panel;
    }

    // ────────────────────────── Micro-helpers

    private HBox metric(String key, String value) {
        Label k = new Label(key + ":  ");
        k.setFont(Font.font("System", FontWeight.BOLD, 13));
        k.setTextFill(Color.web("#8b949e"));

        Label v = new Label(value);
        v.setFont(Font.font("Monospaced", FontWeight.NORMAL, 13));
        v.setTextFill(Color.web("#e6edf3"));

        return new HBox(k, v);
    }

    private Separator separator() {
        Separator s = new Separator();
        s.setStyle("-fx-background-color: #30363d;");
        return s;
    }

    private Color voteColor(TemplateMatcher.MatchResult m) {
        if (m.score >= 0.70) return Color.LIMEGREEN;
        if (m.score >= 0.50) return Color.ORANGE;
        return Color.RED;
    }
}
