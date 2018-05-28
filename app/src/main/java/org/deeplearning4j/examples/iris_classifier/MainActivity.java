package org.deeplearning4j.examples.iris_classifier;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.media.audiofx.NoiseSuppressor;
import android.os.AsyncTask;
import android.os.Process;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.nio.ShortBuffer;
import java.text.DecimalFormat;

import static java.lang.StrictMath.abs;


public class MainActivity extends AppCompatActivity {

    //Global variables to accept the classification results from the background thread.
    double first;
    double second;
    double third;
    double forth;
    double fifth;
    private TextView buffer_count;

    private ShortBuffer mSamples;
    final int SAMPLE_RATE = 44100;
    private static final String TAG = "MyDEBUG";
    private int mNumSamples = SAMPLE_RATE*1;


    private float[][] helperdata = new float[30000][50];
    private float[][] mlabelData = new float[30000][5];
    private int mdatai = 0;

    private TextView iteCountView;
    private int itecounter = 0;

    //public network
    public MultiLayerNetwork myNetwork;

    //Linear predictor
    LinearPredictor linPred;
    private double[] ARParameters;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);




        Button buttonA = (Button) findViewById(R.id.a_button);
        Button buttonB = (Button) findViewById(R.id.b_button);
        Button buttonC = (Button) findViewById(R.id.c_button);
        Button buttonO = (Button) findViewById(R.id.o_button);
        Button buttonU = (Button) findViewById(R.id.u_button);
        Button buttonT = (Button) findViewById(R.id.t_button);

        Button buttonFit = (Button) findViewById(R.id.fit_button);

        iteCountView = (TextView) findViewById(R.id.iterationCount);

        buffer_count = (TextView) findViewById(R.id.buffer_coumt);

        buttonA.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncRecord recorder = new AsyncRecord();
                recorder.execute("a");

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        buttonB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncRecord recorder = new AsyncRecord();
                recorder.execute("e");

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        buttonC.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncRecord recorder = new AsyncRecord();
                recorder.execute("i");

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        buttonO.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncRecord recorder = new AsyncRecord();
                recorder.execute("o");

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        buttonU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncRecord recorder = new AsyncRecord();
                recorder.execute("u");

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        buttonT.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncTrain trainer = new AsyncTrain();
                trainer.execute();

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

        buttonFit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                AsyncFit fitter = new AsyncFit();
                fitter.execute();

                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });


        mSamples = ShortBuffer.allocate(mNumSamples+11111);

      //  linPred = new LinearPredictor(10);
      //  ARParameters = new double[10+1];

        AsyncBuild builder = new AsyncBuild();
        builder.execute();


        }

    private class AsyncBuild extends AsyncTask<Void, Void, Void>{

        @Override
        protected void onPreExecute(){
            super.onPreExecute();

            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.VISIBLE);
        }

        protected Void doInBackground(Void...params) {

            //build the layers of the network
            DenseLayer inputLayer = new DenseLayer.Builder()
                    .nIn(50)
                    .nOut(15)
                    .name("Input")
                    .build();

            DenseLayer hiddenLayer = new DenseLayer.Builder()
                    .nIn(15)
                    .nOut(15)
                    .name("Hidden")
                    .build();

            DenseLayer hiddenLayer2 = new DenseLayer.Builder()
                    .nIn(15)
                    .nOut(15)
                    .name("Hidden2")
                    .build();

            OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(15)
                    .nOut(5)
                    .name("Output")
                    .activation(Activation.SOFTMAX)
                    .build();


            NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
            long seed = 6;
            nncBuilder.seed(seed);
            nncBuilder.iterations(500);
            nncBuilder.learningRate(0.1);
            nncBuilder.activation(Activation.TANH);
            nncBuilder.weightInit(WeightInit.XAVIER);
            nncBuilder.regularization(true).l2(1e-4);

            NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
            listBuilder.layer(0, inputLayer);
            listBuilder.layer(1, hiddenLayer);
            listBuilder.layer(2, hiddenLayer2);
            listBuilder.layer(3, outputLayer);

            listBuilder.backprop(true);

            myNetwork = new MultiLayerNetwork(listBuilder.build());
            myNetwork.init();

            myNetwork.setListeners(new IterationListener() {
                @Override
                public boolean invoked() {

                    return false;
                }

                @Override
                public void invoke() {

                }

                @Override
                public void iterationDone(Model model, int iteration) {
                        itecounter = iteration;
                }
            });




        return null;}

        @Override
        protected void onPostExecute(Void param) {
            super.onPostExecute(null);

            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);

            Toast.makeText(MainActivity.this, "Neural network created", Toast.LENGTH_LONG).show();

        }

    }


    private class AsyncTrain extends AsyncTask<Void, Void, Void> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();

            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.VISIBLE);
        }

        // This is our main background thread for the neural net
        @Override
        protected Void doInBackground(Void...params) {
        //Get the doubles from params, which is an array so they will be 0,1,2,3


        //Create input
       //     INDArray actualInput = Nd4j.zeros(1,4);
         //   actualInput.putScalar(new int[]{0,0}, pld);
        //    actualInput.putScalar(new int[]{0,1}, pwd);
         //   actualInput.putScalar(new int[]{0,2}, sld);
         //   actualInput.putScalar(new int[]{0,3}, swd);


            //Convert the iris data into 150x4 matrix
            //Rescale matrix
            int row=mdatai;
            int col=50;

            float[][] dataMatrix=new float[row][col];

            for(int r=0; r<row; r++){

                for(int c = 0; c<col; c++){
                    dataMatrix[r][c]=helperdata[r][c];

                }


            }


        //Now do the same for the label data
            int rowLabel=mdatai;
            int colLabel=5;

            float[][] twodimLabel=new float[rowLabel][colLabel];

            for(int r=0; r<rowLabel; r++){
                twodimLabel[r][0]=mlabelData[r][0];
                twodimLabel[r][1]=mlabelData[r][1];
                twodimLabel[r][2]=mlabelData[r][2];
                twodimLabel[r][3]=mlabelData[r][3];
                twodimLabel[r][4]=mlabelData[r][4];
            }


        //Convert the data matrices into training INDArrays
            INDArray trainingIn = Nd4j.create(dataMatrix);
            INDArray trainingOut = Nd4j.create(twodimLabel);


        //Create a data set from the INDArrays and train the network
            DataSet myData = new DataSet(trainingIn, trainingOut);
            myNetwork.fit(myData);






        //Evaluate the input data against the model
         //   INDArray actualOutput = myNetwork.output(actualInput);
         //   Log.d("myNetwork Output ", actualOutput.toString());

        //Retrieve the three probabilities
        //    first = actualOutput.getDouble(0,0);
        //    second = actualOutput.getDouble(0,1);
        //    third = actualOutput.getDouble(0,2);

        //Since we used global variables to store the classification results, no need to return
        //a results string. If the results were returned here they would be passed to onPostExecute.
            return null;
        }

        //This is called from background thread but runs in UI for a progress indicator
       // @Override
       // protected Void onProgressUpdate() {
        //    super.onProgressUpdate();
       // }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(Void param) {
            super.onPostExecute(null);

        //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);

        //Update the UI with output
        //    TextView a_prob = (TextView) findViewById(R.id.a_prob);
       //     TextView b_prob = (TextView) findViewById(R.id.b_prob);
       //     TextView c_prob = (TextView) findViewById(R.id.c_prob);

        //Limit the double to values to two decimals using DecimalFormat
         //   DecimalFormat df2 = new DecimalFormat(".##");

        //    a_prob.setText(String.valueOf(df2.format(first)));
        //    b_prob.setText(String.valueOf(df2.format(second)));
        //    c_prob.setText(String.valueOf(df2.format(third)));

            Toast.makeText(MainActivity.this, "Training completed!", Toast.LENGTH_LONG).show();
        }
    }

    private class AsyncFit extends AsyncTask<Void, Void, String>{

        @Override
        protected void onPreExecute(){
            super.onPreExecute();

            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.VISIBLE);
        }

        protected String doInBackground(Void...params) {

            LinearPredictiveCoding LinPred = new LinearPredictiveCoding(1001, 50);
            Normalizer norm = new Normalizer();

            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);

            //buffer size
            int buffersize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);

            if (buffersize == AudioRecord.ERROR || buffersize == AudioRecord.ERROR_BAD_VALUE) {
                buffersize = SAMPLE_RATE * 2;
            }

            short[] audioBuffer = new short[buffersize/2];

            AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    buffersize);

            if (record.getState() != AudioRecord.STATE_INITIALIZED){
                Log.e(TAG,  "Audio Record can't initialize!");
                return "0";
            }

            AudioManager mAudioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            mAudioManager.setMode(AudioManager.MODE_IN_CALL);
            mAudioManager.setParameters("noise_suppresion=on");

            if (NoiseSuppressor.isAvailable()) {
                NoiseSuppressor.create(record.getAudioSessionId());
            } else  Log.d(TAG, "Noise Suppressor is not available");


            record.startRecording();

            Log.v(TAG, "Start recording");
            mSamples.rewind();
            int shortsRead = 0;
            while (shortsRead<mNumSamples){
                int numberOfShort = record.read(audioBuffer, 0 ,audioBuffer.length);
                mSamples.put(audioBuffer);

                shortsRead += numberOfShort;
            }

            record.stop();
            record.release();

            //helperandsave(slovo);
            Log.v(TAG, "Recording stoped");

            Helper hlp = new Helper();
            short tst[] = new short[1001];

            int limit = mNumSamples;
            int totalWriten = 0;
            int iterations = 0;
            int iterationsfull = 0;
            first = 0;
            second = 0;
            third = 0;
            forth = 0;
            fifth = 0;

            mSamples.rewind();
            INDArray actualInput = Nd4j.zeros(1,50);

            while (mSamples.position()< limit) {
                int numSamplesLeft = limit - mSamples.position();
                int samplesToWrite;
                if (numSamplesLeft >= tst.length) {
                    mSamples.get(tst);
                    samplesToWrite = tst.length;
                } else {
                    for (int i = numSamplesLeft; i < tst.length; i++) {
                        tst[i] = 0;
                    }
                    mSamples.get(tst, 0, numSamplesLeft);
                    samplesToWrite = numSamplesLeft;
                }
                totalWriten += samplesToWrite;

                //float sum = hlp.energija(tst);
                float zero = hlp.zcr(tst);

                double[] tstD = hlp.doubleMe(tst);

                double gain = norm.normalize(tstD, 0);

                tstD = hlp.normalization(tstD, gain);

               // int frek = 0;
               // float sum2 = hlp.AutoCorr(tst, frek);

              //  while ((abs(sum2) >= abs(hlp.AutoCorr(tst, frek))) && (frek <750)) {
              //      sum2 = hlp.AutoCorr(tst, frek);
              //      frek ++;
              //  }

             //   float frekf = (float) frek - 1;

               if (zero < 250){

                   double[][] result = LinPred.applyLinearPredictiveCoding(tstD);

                   for (int z = 0; z<50; z++){
                       helperdata[mdatai][z] = (float) result[0][z];

                   }
                    for(int s = 0; s<10; s++){
                        actualInput.putScalar(new int[]{0,s}, result[0][s]);

                    }


                   INDArray actualOutput = myNetwork.output(actualInput);

                   first += actualOutput.getDouble(0,0);
                   second += actualOutput.getDouble(0,1);
                   third += actualOutput.getDouble(0,2);
                   forth += actualOutput.getDouble(0, 3);
                   fifth += actualOutput.getDouble(0,4);

                   iterations++;
               }
                iterationsfull++;
            }

            first = first/iterations;
            second = second/iterations;
            third = third/iterations;
            forth = forth/iterations;
            fifth = fifth/iterations;


            //Evaluate the input data against the model
            //   INDArray actualOutput = myNetwork.output(actualInput);
            //   Log.d("myNetwork Output ", actualOutput.toString());

            //Retrieve the three probabilities
            //    first = actualOutput.getDouble(0,0);
            //    second = actualOutput.getDouble(0,1);
            //    third = actualOutput.getDouble(0,2);

            //Since we used global variables to store the classification results, no need to return
            //a results string. If the results were returned here they would be passed to onPostExecute.

            return "1";}

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);

            if (result.equals("1")){
                Toast.makeText(MainActivity.this, "Analiza gotova", Toast.LENGTH_LONG).show();

                //Update the UI with output
                    TextView a_prob = (TextView) findViewById(R.id.a_prob);
                     TextView b_prob = (TextView) findViewById(R.id.b_prob);
                     TextView c_prob = (TextView) findViewById(R.id.c_prob);
                     TextView o_prob = (TextView) findViewById(R.id.o_prob);
                     TextView u_prob = (TextView) findViewById(R.id.u_prob);

                //Limit the double to values to two decimals using DecimalFormat
                   DecimalFormat df2 = new DecimalFormat(".##");

                    a_prob.setText(String.valueOf(df2.format(first)));
                    b_prob.setText(String.valueOf(df2.format(second)));
                    c_prob.setText(String.valueOf(df2.format(third)));
                      o_prob.setText(String.valueOf(df2.format(forth)));
                      u_prob.setText(String.valueOf(df2.format(fifth)));

            } else {
                Toast.makeText(MainActivity.this, "Greška", Toast.LENGTH_LONG).show();
            }


        }

    }

    private class AsyncRecord extends AsyncTask<String, Void, String>{

        @Override
        protected void onPreExecute(){
            super.onPreExecute();

            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.VISIBLE);
        }

        protected String doInBackground(String...params) {

            String slovo = params[0];
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);

            //buffer size
            int buffersize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);

            if (buffersize == AudioRecord.ERROR || buffersize == AudioRecord.ERROR_BAD_VALUE) {
                buffersize = SAMPLE_RATE * 2;
            }

            short[] audioBuffer = new short[buffersize/2];

            AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    buffersize);

            if (record.getState() != AudioRecord.STATE_INITIALIZED){
                Log.e(TAG,  "Audio Record can't initialize!");
                return "0";
            }

            AudioManager mAudioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            mAudioManager.setMode(AudioManager.MODE_IN_CALL);
            mAudioManager.setParameters("noise_suppresion=on");

            if (NoiseSuppressor.isAvailable()) {
                NoiseSuppressor.create(record.getAudioSessionId());
            } else  Log.d(TAG, "Noise Suppressor is not available");


            record.startRecording();

            Log.v(TAG, "Start recording");
            mSamples.rewind();
            int shortsRead = 0;
            while (shortsRead<mNumSamples){
                int numberOfShort = record.read(audioBuffer, 0 ,audioBuffer.length);
                mSamples.put(audioBuffer);

                shortsRead += numberOfShort;
            }

            record.stop();
            record.release();

            helperandsave(slovo);
            Log.v(TAG, "Recording stoped");

            return "1";}

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);

            String yn = result;
            if (result.equals("1")){
                Toast.makeText(MainActivity.this, "Item recorded!", Toast.LENGTH_LONG).show();
                buffer_count.setText(mdatai + "/20000");
            } else {
                Toast.makeText(MainActivity.this, "Error!", Toast.LENGTH_LONG).show();
            }


        }

    }

    //deprecated
    public void record(View view){

      new Thread(new Runnable() {
            @Override
            public void run() {


                Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);

                //buffer size
                int buffersize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);

                if (buffersize == AudioRecord.ERROR || buffersize == AudioRecord.ERROR_BAD_VALUE) {
                    buffersize = SAMPLE_RATE * 2;
                }

                short[] audioBuffer = new short[buffersize/2];

                AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        buffersize);

                if (record.getState() != AudioRecord.STATE_INITIALIZED){
                    Log.e(TAG,  "Audio Record can't initialize!");
                    return;
                }

                AudioManager mAudioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
                mAudioManager.setMode(AudioManager.MODE_IN_CALL);
                mAudioManager.setParameters("noise_suppresion=on");

                if (NoiseSuppressor.isAvailable()) {
                    NoiseSuppressor.create(record.getAudioSessionId());
                } else  Log.d(TAG, "Noise Suppressor is not available");


                record.startRecording();

                Log.v(TAG, "Start recording");
                mSamples.rewind();
                int shortsRead = 0;
                while (shortsRead<mNumSamples){
                    int numberOfShort = record.read(audioBuffer, 0 ,audioBuffer.length);
                    mSamples.put(audioBuffer);

                    shortsRead += numberOfShort;
                }

                record.stop();
                record.release();

                helperandsave("a");
                Log.v(TAG, "Recording stoped");


            }
        }).start();

    }

    public void helperandsave(String slovo){
        Helper hlp = new Helper();
        short tst[] = new short[1001];

        Normalizer norm = new Normalizer();

        LinearPredictiveCoding LinPred = new LinearPredictiveCoding(1001, 50);

        int limit = mNumSamples;
        int totalWriten = 0;

        mSamples.rewind();
        int iter = 0;

        while (mSamples.position()< limit) {
            int numSamplesLeft = limit - mSamples.position();
            int samplesToWrite;
            if (numSamplesLeft>= tst.length) {
                mSamples.get(tst);
                samplesToWrite = tst.length;
            } else {
                for (int i = numSamplesLeft; i < tst.length; i++){
                    tst[i]= 0;
                }
                mSamples.get(tst, 0 , numSamplesLeft);
                samplesToWrite = numSamplesLeft;
            }
            totalWriten += samplesToWrite;

           // float sum = hlp.energija(tst);
            float zero = hlp.zcr(tst);

            double[] tstD = hlp.doubleMe(tst);

            double gain = norm.normalize(tstD, 0);

           tstD = hlp.normalization(tstD, gain);
            //tstD = norm.normalize(tstD, 0);
           // ARParameters = linPred.getARFilter()

          //  int frek = 0;
          //  float sum2 = hlp.AutoCorr(tst, frek);

         //   while ((abs(sum2) >= abs(hlp.AutoCorr(tst, frek))) && (frek <1000)) {
         //       sum2 = hlp.AutoCorr(tst, frek);
         //       frek ++;
         //   }

         //   float frekf = (float) frek - 1;



            //TODO:ako ima vise od 1000ZCR zanemari
            if (zero < 250) {

                double[][] result = LinPred.applyLinearPredictiveCoding(tstD);

                for (int z = 0; z<50; z++){
                    helperdata[mdatai][z] = (float) result[0][z];
                }
           //     helperdata[mdatai][0] = sum;
            //    helperdata[mdatai][1] = zero;
           //     helperdata[mdatai][2] = frekf;

                switch (slovo) {
                    case "a": mlabelData[mdatai][0]=1;
                        mlabelData[mdatai][1] = 0;
                        mlabelData[mdatai][2] = 0;
                        mlabelData[mdatai][3] = 0;
                        mlabelData[mdatai][4] = 0;
                        break;
                    case "e": mlabelData[mdatai][0]=0;
                        mlabelData[mdatai][1] = 1;
                        mlabelData[mdatai][2] = 0;
                        mlabelData[mdatai][3] = 0;
                        mlabelData[mdatai][4] = 0;
                        break;
                    case "i": mlabelData[mdatai][0]=0;
                        mlabelData[mdatai][1] = 0;
                        mlabelData[mdatai][2] = 1;
                        mlabelData[mdatai][3] = 0;
                        mlabelData[mdatai][4] = 0;
                        break;
                    case "o": mlabelData[mdatai][0]=0;
                        mlabelData[mdatai][1] = 0;
                        mlabelData[mdatai][2] = 0;
                        mlabelData[mdatai][3] = 1;
                        mlabelData[mdatai][4] = 0;
                        break;
                    case "u": mlabelData[mdatai][0]=0;
                        mlabelData[mdatai][1] = 0;
                        mlabelData[mdatai][2] = 0;
                        mlabelData[mdatai][3] = 0;
                        mlabelData[mdatai][4] = 1;
                        break;
                    default: break;
                }
                mdatai++;
            }

        iter++;
        }
        Log.v("TAG", "Test");


    }



    //deprecated
    public void play(View view){

        new Thread(new Runnable() {
            @Override
            public void run() {
                int bufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);

                if (bufferSize == AudioTrack.ERROR || bufferSize == AudioTrack.ERROR_BAD_VALUE){
                    bufferSize = SAMPLE_RATE*2;
                }

                AudioTrack audioTrack = new AudioTrack(
                        AudioManager.STREAM_MUSIC,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize,
                        AudioTrack.MODE_STREAM);

                audioTrack.play();

                Log.v(TAG, "Streaming audio");

                short[] buffer = new short[bufferSize];
                mSamples.rewind();
                int limit = mNumSamples;
                int totalWriten = 0;

                while (mSamples.position()< limit) {
                    int numSamplesLeft = limit - mSamples.position();
                    int samplesToWrite;
                    if (numSamplesLeft>= buffer.length) {
                        mSamples.get(buffer);
                        samplesToWrite = buffer.length;
                    } else {
                        for (int i = numSamplesLeft; i < buffer.length; i++){
                            buffer[i]= 0;
                        }
                        mSamples.get(buffer, 0 , numSamplesLeft);
                        samplesToWrite = numSamplesLeft;
                    }
                    totalWriten += samplesToWrite;
                    audioTrack.write(buffer, 0 , samplesToWrite);
                }

                audioTrack.release();

                Log.v(TAG, "Audio streaming finished. Samples writen: " + totalWriten);
            }
        }).start();
    }

    public void refreshIte(View view){
        iteCountView.setText(String.valueOf(itecounter));

    }
}
