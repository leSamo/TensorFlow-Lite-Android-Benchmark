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

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.OptionalDouble;
import java.util.stream.*;

public class Benchmark extends Activity {
    private static final Logger LOGGER = new Logger();

    private Context context;
    private String modelName;
    private ProgressDialog dialog;
    private int imageId = 0;
    private AssetManager assetManager;
    String[] files;
    int[] inferenceTimes;

    final String TEST_FOLDER = "benchmark/";

    public Benchmark(Context context, String modelName) throws IOException {
        this.context = context;
        this.modelName = modelName;

        assetManager = context.getAssets();
        files = context.getAssets().list(TEST_FOLDER);

        dialog = ProgressDialog.show(context, "Running benchmark", "Classifying image 1/" + files.length, true);

        inferenceTimes = new int[files.length];
    }

    public Bitmap getNextBitmap() throws IOException {
        if (imageId >= files.length) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    dialog.hide();
                }
            });

            showResultDialog();

            return null;
        }

        dialog.setMessage("Classifying image " + (imageId + 1) + "/" + files.length);
        LOGGER.e("Loading " + imageId);

        InputStream istr = assetManager.open(TEST_FOLDER + files[imageId]);
        Bitmap bitmap = BitmapFactory.decodeStream(istr);
        istr.close();

        imageId++;

        return bitmap;
    }

    public void logTime(int ms) {
        if (imageId < files.length) {
            inferenceTimes[imageId] = ms;
        }
    }

    private void showResultDialog() {
        double averageInferenceTime = IntStream.of(inferenceTimes).average().orElse(Double.NaN);

        new AlertDialog.Builder(context)
            .setTitle("Benchmark results")
            .setMessage(
                "Device: " + getDeviceName() + "\n"
                + "Android version: " + Build.VERSION.RELEASE + "\n"
                + "Model: " + modelName + "\n"
                + "Input resolution: 224x224" + "\n"
                + "Average inference time: " + String.format(Locale.ENGLISH, "%.2f", averageInferenceTime) + "ms" + "\n"
                + "Average FPS: " + String.format(Locale.ENGLISH, "%.2f", 1 / averageInferenceTime * 1000) + "\n"
                + "Variance: " + String.format(Locale.ENGLISH, "%.2f", getVariance()) + "ms^2"
            )
            .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                public void onClick(DialogInterface dialog, int which) {
                    // Continue with delete operation
                }
            })
            .setIcon(android.R.drawable.ic_dialog_info)
            .show();
    }

    private double getVariance() {
        // The mean average
        double mean = 0.0;

        for (int i = 0; i < inferenceTimes.length; i++) {
            mean += inferenceTimes[i];
        }
        mean /= inferenceTimes.length;

        // The variance
        double variance = 0;

        for (int i = 0; i < inferenceTimes.length; i++) {
            variance += Math.pow(inferenceTimes[i] - mean, 2);
        }
        variance /= inferenceTimes.length;

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
