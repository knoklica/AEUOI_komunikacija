package org.deeplearning4j.examples.iris_classifier;

public class Helper {

    public float energija(short s[]){
        float sum = 0;

        for(float ss: s){
            sum+= (ss/65535)*(ss/65535);
        }

        return sum;
    }

    public float zcr(short s[]){
        float br_prijelaz = 0;

        for(int i=1; i<s.length;i++) {
            if (s[i - 1] * s[i] <= 0) {
                br_prijelaz++;
            }
        }
        return br_prijelaz;
    }

    public float AutoCorr(short s[], int k){
        float sum = 0;

        for (int i =0; i<s.length-k ; i++){
            sum +=((float) s[i]/65535)*((float)s[i+k]/65535);
        }

        return sum;
    }

    public double[] doubleMe(short[] pcms) {
        double[] doubles = new double[pcms.length];
        for (int i = 0; i < pcms.length; i++) {
            doubles[i] = pcms[i] / 32768.0;
        }
        return doubles;
    }

    public double[] normalization (double[] audio, double gain) {
        double[] temp = new double[audio.length];
        for (int i = 0; i< audio.length; i++){
            temp[i] = audio[i] / gain;
        }
        return temp;
    }


}
