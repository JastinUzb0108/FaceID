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

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Pair;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel
        implements SimilarityClassifier {

  private static final Logger LOGGER = new Logger();

 // private static final int OUTPUT_SIZE = 512;
  private static final int OUTPUT_SIZE = 192;

  // Only return this many results.
  private static final int NUM_DETECTIONS = 1;

  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;

  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private float[][] embeedings;

  private ByteBuffer imgData;

  private Interpreter tfLite;

// Face Mask Detector Output
  private float[][] output;
  public int getUserCount(){
    return registered.size();
  }
public void readFaces(){
//int a=0;
//if(a==0)return;
  Recognition myRec=new Recognition("100","myTitle",(float)0.0,null);
  double[] myArray={0.0031208172,-6.873479E-4,0.0069750925,-0.009994449,-0.006336254,-0.022192895,0.049928043,0.18175764,-0.06759258,0.1189538,-0.004216269,-0.0028703853,-6.142958E-4,-0.01100156,-7.69981E-4,-0.031735294,-0.007963418,-0.0064666793,-0.0017188409,1.3200631E-5,0.21259648,-0.017452043,-0.015494936,-7.915714E-4,0.121475846,0.008524572,-0.005895779,0.050661992,0.09524358,-0.1393338,0.011684674,0.089372575,-0.19362637,-0.011667956,-0.24824467,-0.053739328,0.25116122,0.019128712,6.059211E-4,-0.3637321,2.9157865E-4,0.0014252467,-0.0019964995,-0.004250992,0.0068542506,-1.12976915E-4,-0.09543145,0.08094086,-0.0021502918,0.006941572,0.122385845,-0.0011681087,-0.009905592,-0.001226709,-0.036517955,0.0101516135,0.11741621,-0.001518678,-4.5642228E-5,-0.006548158,5.0261855E-4,0.0025085523,-0.01913341,-0.029457554,-0.0012848035,-0.1425309,-0.004391574,0.0027596168,0.015312293,0.001435966,-0.006625717,0.007542281,0.0013307496,0.0012378296,0.08163454,0.0026559548,0.0026052103,-2.5195544E-4,-0.10663096,-0.028352696,0.006430122,-0.03130114,-0.007587753,0.13711625,0.1050883,0.005990454,0.011000756,-0.0028683569,-0.035288896,0.26459664,0.14050484,-0.010163747,-0.011614271,-0.013483465,-0.054435533,-0.17963225,0.043690108,0.03341945,0.002759592,-0.021023616,0.001903333,-5.77944E-4,0.009164487,0.0030854498,0.0017264981,0.0040973336,0.027179234,0.0013089979,0.0012679683,-0.0050176024,0.18138194,-7.161481E-4,0.0028063154,-0.030352592,0.0059902705,-0.084774815,-0.0052336794,-0.03186444,-0.08888011,-0.020191079,0.10705513,-0.0035395669,0.13274777,-7.713376E-4,0.0030864675,-0.0023596105,0.007555048,0.0014820957,0.019263845,0.029303098,-0.0011918569,0.0017741274,-0.0031034497,-0.04644421,0.07283737,-0.0028941575,-0.0056549245,0.041915495,0.0031197206,0.014797761,0.0047171814,-0.013832866,0.002594885,0.07253675,0.008652339,0.17003106,-0.029985925,6.4111396E-4,0.0026861194,0.011211514,0.01300834,-0.14880626,0.102525115,0.0036359834,0.002667132,0.0043841423,0.003953344,9.065239E-4,0.09222664,0.0081789605,-0.01023454,0.003963145,-0.004088504,-6.991456E-4,0.001442523,9.279499E-4,-4.7497215E-5,0.21296462,-0.0010106071,-0.0023511522,0.040917855,-0.0388305,0.004474566,-0.06022361,-0.053662427,0.0026621376,0.030443326,0.032092776,-0.0029273042,-0.0015595033,-0.13271415,0.068682455,-0.005260158,0.0037947942,-0.06450127,-0.07787309,-0.085280225,-0.046243493,0.1643738,-0.06000227,-0.005668649,0.0045796223};
  embeedings = new float[1][OUTPUT_SIZE];
  for(int i=0;i<myArray.length;i++) {
    embeedings[0][i] = (float) myArray[i];
  }
  myRec.setExtra(embeedings);
  registered.put("SHOHRUH", myRec);

  myRec=new Recognition("100","myTitle",(float)0.0,null);
  double[] myArray1={0.002319772,-0.0010732118,0.0033815377,0.0011068828,-0.008159216,-0.06255621,-0.03030844,-0.13912597,0.11655853,-0.14631459,-0.008772413,0.0053789276,-0.0013028785,2.1949002E-4,0.002559525,-0.03307088,-0.007911748,0.0012411717,-0.001487595,0.0058163153,-0.07219431,0.0070015327,0.18668593,7.4423786E-4,0.108566515,0.019067822,-0.003313753,-0.029156042,-0.2621402,0.078736626,0.01474732,0.2256564,0.06718125,-1.6774646E-4,0.18864729,0.15573877,-0.042995624,-0.0016806549,0.0055558607,0.15611993,-4.6826547E-4,0.0017387839,-0.0012435645,0.0039730035,-0.004495436,0.011432275,0.10173236,-0.024477044,-0.0021530676,0.045149848,0.04469534,-0.0034426528,0.1622931,0.0019708977,-0.013741638,-0.00919444,0.043391127,-4.9037545E-4,0.016903803,-0.0025071197,0.040033743,0.10305647,-0.1131202,0.24522929,1.5301823E-4,0.08958667,-0.0027833972,-0.034102272,0.0011802454,0.005548985,0.010331039,0.19453567,-0.06237232,-0.0013610173,0.022753332,0.02529061,0.010623637,0.0017774228,-0.0065763616,-0.09420782,-0.0053858464,0.1028414,0.008847474,0.066788666,0.009658365,-0.0022863785,-0.0047480953,0.023157025,0.028826939,0.09378608,0.055732425,-0.0014003873,-0.006156952,-0.0041502607,-0.18950622,-0.21876046,0.09389726,0.0527637,-0.0040349467,-0.015360374,0.0026830996,0.0032281938,-4.770788E-4,-4.7695937E-4,0.008558204,0.0031246436,0.04111259,-0.0024070027,0.01176678,-0.0050676027,0.22469194,0.002779987,-0.007627817,-0.11395325,0.0053660376,-0.009623583,-0.0095018875,0.022733226,-0.084676,-0.05290824,0.3263748,-0.028984258,0.08117032,-0.0068722293,0.001698545,0.0065439,0.005107019,0.00727,-0.019367974,-0.13631171,-7.3264894E-4,0.013167898,-0.0035880446,0.0049346695,0.0035375468,-0.0058952025,-0.017854411,0.01601188,0.008102711,0.011645927,-0.0068128183,-0.006710147,-0.0016235802,0.0064683356,0.06931813,0.124075666,6.3292996E-4,0.013504472,0.010943368,0.0018200683,-0.0017405959,-0.01989776,0.03964599,-0.0046300692,-0.0053846017,-0.017067371,0.005570144,0.0010564065,0.09007148,0.0019308956,-0.016944766,0.0034058671,0.002971196,-5.8109156E-4,-0.0057441737,0.0040318416,-0.0055732704,0.052920204,-8.945219E-4,-0.0011520932,-0.11086873,0.046242267,0.0038356737,0.12426418,0.020621046,-0.0014056738,0.10820462,-0.011140353,-0.0044188993,0.003835834,-0.065711476,0.023635227,-0.009854564,0.0038784083,-0.09912526,0.07328348,-0.06156622,0.034493368,-0.20553839,-0.027418498,-0.0026334785,0.00911898};
  embeedings = new float[1][OUTPUT_SIZE];
  for(int i=0;i<embeedings[0].length;i++) {
    embeedings[0][i] = (float) myArray1[i];
  }
  myRec.setExtra(embeedings);
  registered.put("G'AYRAT", myRec);



  for(int k=0;k<9997;k++){
    embeedings = new float[1][OUTPUT_SIZE];
    myRec=new Recognition("100","myTitle",(float)0.0,null);
    Random r = new Random();
    for(int i=0;i<embeedings[0].length;i++) {
      embeedings[0][i] = (float) (-0.3 + r.nextFloat() * (0.1 + 0.3)+myArray[i]);
    }
    myRec.setExtra(embeedings);
    registered.put("user"+k, myRec);
  }
  myRec=new Recognition("100","myTitle",(float)0.0,null);
  double[] myArray2={5.4570363E-4,-3.6016342E-4,0.012287093,4.7470923E-4,-0.0067612818,-0.088653326,0.07682998,0.104793414,-0.0019970543,0.42214608,0.0011107777,0.0038326613,-0.0040445346,0.007937025,0.0016575275,0.08172995,0.0045371787,0.0070734364,-0.0058808695,-0.0050965045,0.029268874,0.034621723,0.15967964,-0.0070900004,-0.0076073604,0.0016048905,0.004259293,0.16745837,0.16974227,-0.17081666,-0.019918442,-0.19414026,-0.1473456,0.0066135144,0.21131635,0.093114704,0.15149349,-0.03093675,0.009528277,-0.30903155,-0.0071127266,-7.4255426E-4,2.9942236E-4,0.0034970702,0.016564196,-0.02149853,0.09355183,0.04631442,0.0030862405,-0.03173733,-0.03368013,-3.3873616E-4,0.044043258,0.004749294,0.02154492,0.009646728,0.04308936,-0.001501279,-0.024527647,-0.00481351,0.023761468,-0.0865449,-0.09249685,-0.19398682,0.015222576,-0.01837529,0.0012724506,-0.014808903,9.110786E-4,-0.004308673,-0.00821407,0.07436887,-0.022733387,5.6062033E-4,-0.003618492,-0.007413866,0.013530092,-0.0023883276,-0.024956396,-0.006234836,-0.0065635606,-8.0715586E-4,0.004507172,0.06652238,0.10797396,-0.0020549593,0.013872838,0.014107275,-0.033564314,-0.05268463,-0.040605877,-0.015274858,0.0044370755,0.027282296,-0.018766973,-0.094654635,-0.12052215,0.15605873,-0.0029258118,-0.014775379,-0.001268419,-0.006129776,0.005821654,0.005514421,-0.0092546195,-9.785737E-4,-0.085849866,0.0040685683,-2.8668236E-4,-0.008686661,0.06901533,-0.0064964243,-0.0011788688,-0.16004942,0.0030068439,0.18929152,-0.018480271,1.2450812E-5,-0.22556196,-0.097629294,0.12952933,0.030154098,-0.09393061,-0.0015000586,0.0045286375,-0.0022699789,-0.0040366254,0.0013804792,-0.0053945533,-0.014130163,-0.0030262189,-0.0060283053,-0.002559392,0.03906926,0.02745466,-0.0041767084,-0.024203165,-0.038293764,-0.0027948008,0.008296079,-0.009427372,0.0031246028,-0.007062903,0.021490142,0.013334646,0.0791841,0.0070556137,-0.0034626143,0.004005821,-0.003388565,0.017701972,-0.03993513,0.19812877,0.0038954138,-0.009269984,-0.008296744,7.836156E-4,-0.010589764,-0.053912174,-0.0012172613,-0.021162802,-5.974896E-4,-0.0059030973,5.4955296E-4,-0.0062132063,0.004443484,0.008741534,-0.11314357,0.0039763274,0.0035156563,0.055302564,-0.12799439,-0.0012678,0.031267505,-0.014271202,-2.86552E-4,-0.053350814,-0.018889459,-0.0019796656,-0.0035021505,-0.043809902,-0.1265558,-0.008277429,0.0020950632,-0.004007271,-0.088117026,-0.036224302,0.003178354,-0.090246476,0.032516282,-0.008658334,0.0010593145}; embeedings = new float[1][OUTPUT_SIZE];
  embeedings = new float[1][OUTPUT_SIZE];
  for(int i=0;i<embeedings[0].length;i++) {
    embeedings[0][i] = (float) myArray2[i];
  }
  myRec.setExtra(embeedings);
  registered.put("UMRBEK", myRec);

  LOGGER.i("MyActivity registered.size():"+ registered.size());
}
  private HashMap<String, Recognition> registered = new HashMap<>();
  public void register(String name, Recognition rec) {
    //
    registered.put(name, rec);


    final float[] knownEmb = ((float[][]) rec.getExtra())[0];
    String str="";
      for(int i=0;i<knownEmb.length;i++)
      str+=knownEmb[i]+",";
      LOGGER.i("MyActivity "+ str);
  }

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static SimilarityClassifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {

    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];
    return d;
  }

  // looks for the nearest embeeding in the dataset (using L2 norm)
  // and retrurns the pair <id, distance>
  private Pair<String, Float> findNearest(float[] emb) {

    Pair<String, Float> ret = null;
    for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
        final String name = entry.getKey();
        final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

        float distance = 0;
        for (int i = 0; i < emb.length; i++) {
              float diff = emb[i] - knownEmb[i];
              distance += diff*diff;
        }
        distance = (float) Math.sqrt(distance);
        if (ret == null || distance < ret.second) {
            ret = new Pair<>(name, distance);
        }
    }

    return ret;

  }


  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap, boolean storeExtra) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");


    Object[] inputArray = {imgData};

    Trace.endSection();

// Here outputMap is changed to fit the Face Mask detector
    Map<Integer, Object> outputMap = new HashMap<>();

    embeedings = new float[1][OUTPUT_SIZE];
    outputMap.put(0, embeedings);


    // Run the inference call.
    Trace.beginSection("run");
    //tfLite.runForMultipleInputsOutputs(inputArray, outputMapBack);
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

//    String res = "[";
//    for (int i = 0; i < embeedings[0].length; i++) {
//      res += embeedings[0][i];
//      if (i < embeedings[0].length - 1) res += ", ";
//    }
//    res += "]";


    float distance = Float.MAX_VALUE;
    String id = "0";
    String label = "?";

    if (registered.size() > 0) {
        //LOGGER.i("dataset SIZE: " + registered.size());
        final Pair<String, Float> nearest = findNearest(embeedings[0]);
        if (nearest != null) {

            final String name = nearest.first;
            label = name;
            distance = nearest.second;

            LOGGER.i("nearest: " + name + " - distance: " + distance);


        }
    }


    final int numDetectionsOutput = 1;
    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    Recognition rec = new Recognition(
            id,
            label,
            distance,
            new RectF());

    recognitions.add( rec );

    if (storeExtra) {
        rec.setExtra(embeedings);
    }

    Trace.endSection();
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
