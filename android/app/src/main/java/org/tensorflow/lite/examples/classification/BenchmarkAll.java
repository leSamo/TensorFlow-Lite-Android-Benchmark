package org.tensorflow.lite.examples.classification;


import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.widget.TextView;

import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.stream.IntStream;

public class BenchmarkAll extends Activity {
    private static final Logger LOGGER = new Logger();

    private Context context;
    private ProgressDialog dialog;
    private int imageId = 0;
    private int networkCount;
    private int networkIndex = 0;
    private AssetManager assetManager;
    String[] files;
    int[][] inferenceTimes;

    final String TEST_FOLDER = "benchmark/";

    public BenchmarkAll(Context context) throws IOException {
        this.context = context;
        this.networkCount = Classifier.Model.values().length;

        assetManager = context.getAssets();
        files = context.getAssets().list(TEST_FOLDER);

        dialog = ProgressDialog.show(context, "Running benchmark", "Network 1/" + networkCount + "\nClassifying image 1/" + files.length, true);

        inferenceTimes = new int[networkCount][files.length];
    }

    public Bitmap getNextBitmap() throws IOException {
        if (imageId >= files.length) {
            return null;
        }

        dialog.setMessage("Network " + (networkIndex + 1) + "/" + networkCount + "\nClassifying image " + (imageId + 1) + "/" + files.length);
        LOGGER.e("Loading " + networkIndex + ":" + imageId);

        InputStream istr = assetManager.open(TEST_FOLDER + files[imageId]);
        Bitmap bitmap = BitmapFactory.decodeStream(istr);
        istr.close();

        imageId++;

        return bitmap;
    }

    public boolean nextNetwork() {
        if (networkIndex >= networkCount - 1) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    dialog.hide();
                }
            });

            showResultDialog();
            return false;
        }
        else {
            imageId = 0;
            networkIndex++;
            return true;
        }
    }

    public void logTime(int ms) {
        if (imageId < files.length && networkIndex < networkCount) {
            inferenceTimes[networkIndex][imageId] = ms;
        }
    }

    private void showResultDialog() {
        String table = "";
        Classifier.Model model = Classifier.Model.MOBILENETV2_96_Q0_M;

        for (int i = 0; i < networkCount; i++) {
            double averageInferenceTime = IntStream.of(inferenceTimes[i]).average().orElse(Double.NaN);

            table += model.values()[i].toString() + "; " + String.format(Locale.ENGLISH, "%.2f", averageInferenceTime) + "; " + String.format(Locale.ENGLISH, "%.2f", getVariance(inferenceTimes[i])) + "\n";
        }

        AlertDialog resultDialog = new AlertDialog.Builder(context)
                .setTitle("Benchmark results")
                .setMessage(
                        "Device: " + getDeviceName() + "\n"
                                + "Android version: " + Build.VERSION.RELEASE + "\n\n"
                                + "model; avg time (ms); variance (ms^2)\n"
                                + table
                )
                .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        // Continue with delete operation
                    }
                })
                .setIcon(android.R.drawable.ic_dialog_info)
                .show();

        TextView textView = resultDialog.findViewById(android.R.id.message);
        textView.setTextSize(12);
    }

    private double getVariance(int[] times) {
        // The mean average
        double mean = 0.0;

        for (int i = 0; i < times.length; i++) {
            mean += times[i];
        }
        mean /= times.length;

        // The variance
        double variance = 0;

        for (int i = 0; i < times.length; i++) {
            variance += Math.pow(times[i] - mean, 2);
        }
        variance /= times.length;

        return variance;
    }

    private String getDeviceName() {
        String manufacturer = Build.MANUFACTURER;
        String model = Build.MODEL;
        if (model.toLowerCase().startsWith(manufacturer.toLowerCase())) {
            return capitalize(model);
        } else {
            return capitalize(manufacturer) + " " + model;
        }
    }

    private String capitalize(String s) {
        if (s == null || s.length() == 0) {
            return "";
        }
        char first = s.charAt(0);
        if (Character.isUpperCase(first)) {
            return s;
        } else {
            return Character.toUpperCase(first) + s.substring(1);
        }
    }
}
