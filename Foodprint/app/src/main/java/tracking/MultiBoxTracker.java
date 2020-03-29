/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tracking;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 18;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.CYAN,
    Color.MAGENTA,
    Color.WHITE,
    Color.parseColor("#55FF55"),
    Color.parseColor("#FFA500"),
    Color.parseColor("#FF8888"),
    Color.parseColor("#AAAAFF"),
    Color.parseColor("#FFFFAA"),
    Color.parseColor("#55AAAA"),
    Color.parseColor("#AA33AA"),
    Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;

  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(10.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
      final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(
            canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
            canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);
    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      final String labelString =
          !TextUtils.isEmpty(recognition.title)
              ? String.format("%s %.2f", carbon(recognition.title), (100 * recognition.detectionConfidence))
              : String.format("%.2f", (100 * recognition.detectionConfidence));
      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);
      borderedText.drawText(
          canvas, trackedPos.left + cornerSize, trackedPos.top, labelString + "%", boxPaint);
    }
  }

  private void processResults(final List<Recognition> results) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.color = COLORS[trackedObjects.size()];
      trackedObjects.add(trackedRecognition);

      if (trackedObjects.size() >= COLORS.length) {
        break;
      }
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    int color;
    String title;
  }

  private String carbon(String input) {
    String result;
    switch (input){
      case "giraffe":
        result= input + " 100" + " Co2e";
        break;
      case "backpack":
        result= input + " 100" + " Co2e";
        break;
      case "umbrella":
        result= input + " 100" + " Co2e";
        break;
      case "handbag":
        result= input + " 100" + " Co2e";
        break;
      case "tie":
        result= input + " 100" + " Co2e";
        break;
      case "suitcase":
        result= input + " 100" + " Co2e";
        break;
      case "frisbee":
        result= input + " 100" + " Co2e";
        break;
      case "skis":
        result= input + " 100" + " Co2e";
        break;
      case "snowboard":
        result= input + " 100" + " Co2e";
        break;
      case "sports":
        result= input + " 100" + " Co2e";
        break;
      case "ball":
        result= input + " 100" + " Co2e";
        break;
      case "kite":
        result= input + " 100" + " Co2e";
        break;
      case "baseball":
        result= input + " 100" + " Co2e";
        break;
      case "bat":
        result= input + " 100" + " Co2e";
        break;
      case "glove":
        result= input + " 100" + " Co2e";
        break;
      case "skateboard":
        result= input + " 100" + " Co2e";
        break;
      case "surfboard":
        result= input + " 100" + " Co2e";
        break;
      case "tennis":
        result= input + " 100" + " Co2e";
        break;
      case "racket":
        result= input + " 100" + " Co2e";
        break;
      case "bottle":
        result= input + " 100" + " Co2e";
        break;
      case "wine":
        result= input + " 100" + " Co2e";
        break;
      case "glass":
        result= input + " 100" + " Co2e";
        break;
      case "cup":
        result= input + " 100" + " Co2e";
        break;
      case "fork":
        result= input + " 100" + " Co2e";
        break;
      case "knife":
        result= input + " 100" + " Co2e";
        break;
      case "spoon":
        result= input + " 100" + " Co2e";
        break;
      case "bowl":
        result= input + " 100" + " Co2e";
        break;
      case "banana":
        result= input + "  0.2734" + " Co2e";
        break;
      case "apple":
        result= input + "  0.2347" + " Co2e";
        break;
      case "sandwich":
        result= input + " 100" + " Co2e";
        break;
      case "orange":
        result= input + " 0.0972" + " Co2e";
        break;
      case "broccoli":
        result= input + " 100" + " Co2e";
        break;
      case "carrot":
        result= input + " 100" + " Co2e";
        break;
      case "hot dog":
        result= input + " 100" + " Co2e";
        break;
      case "pizza":
        result= input + " 100" + " Co2e";
        break;
      case "donut":
        result= input + " 100" + " Co2e";
        break;
      case "cake":
        result= input + " 100" + " Co2e";
        break;
      case "chair":
        result= input + " 100" + " Co2e";
        break;
      case "couch":
        result= input + " 100" + " Co2e";
        break;
      case "???":
        result= input + " 100" + " Co2e";
        break;
      case "person":
        result= input + " 100" + " Co2e";
        break;
      case "bicycle":
        result= input + " 100" + " Co2e";
        break;
      case "car":
        result= input + " 100" + " Co2e";
        break;
      case "motorcycle":
        result= input + " 100" + " Co2e";
        break;
      case "airplane":
        result= input + " 100" + " Co2e";
        break;
      case "bus":
        result= input + " 100" + " Co2e";
        break;
      case "train":
        result= input + " 100" + " Co2e";
        break;
      case "truck":
        result= input + " 100" + " Co2e";
        break;
      case "boat":
        result= input + " 100" + " Co2e";
        break;
      case "traffic":
        result= input + " 100" + " Co2e";
        break;
      case "light":
        result= input + " 100" + " Co2e";
        break;
      case "fire":
        result= input + " 100" + " Co2e";
        break;
      case "hydrant":
        result= input + " 100" + " Co2e";
        break;
      case "stop":
        result= input + " 100" + " Co2e";
        break;
      case "sign":
        result= input + " 100" + " Co2e";
        break;
      case "parking":
        result= input + " 100" + " Co2e";
        break;
      case "meter":
        result= input + " 100" + " Co2e";
        break;
      case "bench":
        result= input + " 100" + " Co2e";
        break;
      case "bird":
        result= input + " 100" + " Co2e";
        break;
      case "cat":
        result= input + " 100" + " Co2e";
        break;
      case "dog":
        result= input + " 100" + " Co2e";
        break;
      case "horse":
        result= input + " 100" + " Co2e";
        break;
      case "sheep":
        result= input + " 100" + " Co2e";
        break;
      case "cow":
        result= input + " 100" + " Co2e";
        break;
      case "elephant":
        result= input + " 100" + " Co2e";
        break;
      case "bear":
        result= input + " 100" + " Co2e";
        break;
      case "zebra":
        result= input + " 100" + " Co2e";
        break;
      case "potted plant":
        result= input + " 100" + " Co2e";
        break;
      case "bed":
        result= input + " 100" + " Co2e";
        break;
      case "dining table":
        result= input + " 100" + " Co2e";
        break;
      case "toilet":
        result= input + " 100" + " Co2e";
        break;
      case "tv":
        result= input + " 215" + " Co2e";
        break;
      case "laptop":
        result= input + " 350" + " Co2e";
        break;
      case "mouse":
        result= input + " 100" + " Co2e";
        break;
      case "remote":
        result= input + " 100" + " Co2e";
        break;
      case "keyboard":
        result= input + " 100" + " Co2e";
        break;
      case "cell phone":
        result= input + " 100" + " Co2e";
        break;
      case "microwave":
        result= input + " 39" +
                "" + " Co2e";
        break;
      case "oven":
        result= input + " 100" + " Co2e";
        break;
      case "toaster":
        result= input + " 100" + " Co2e";
        break;
      case "sink":
        result= input + " 100" + " Co2e";
        break;
      case "refrigerator":
        result= input + " 100" + " Co2e";
        break;
      case "book":
        result= input + " 100" + " Co2e";
        break;
      case "clock":
        result= input + " 100" + " Co2e";
        break;
      case "vase":
        result= input + " 100" + " Co2e";
        break;
      case "scissors":
        result= input + " 100" + " Co2e";
        break;
      case "teddy bear":
        result= input + " 100" + " Co2e";
        break;
      case "hair drier":
        result= input + " 100" + " Co2e";
        break;
      case "toothbrush":
        result= input + " 100" + " Co2e";
        break;
      default:
        result= "";
    }
    return result;
  }
}

