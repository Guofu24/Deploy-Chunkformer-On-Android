








//package com.example.speechtotextdeploy;
//
//import androidx.appcompat.app.AppCompatActivity;
//import androidx.core.app.ActivityCompat;
//import androidx.core.content.ContextCompat;
//
//import android.Manifest;
//import android.content.Context;
//import android.content.pm.PackageManager;
//import android.media.AudioFormat;
//import android.media.AudioRecord;
//import android.media.MediaRecorder;
//import android.os.Build;
//import android.os.Bundle;
//import android.util.Log;
//import android.widget.Button;
//import android.widget.TextView;
//import android.widget.Toast;
//
//import org.jtransforms.fft.FloatFFT_1D;
//import org.pytorch.IValue;
//import org.pytorch.LiteModuleLoader;
//import org.pytorch.Module;
//import org.pytorch.Tensor;
//
//import java.io.BufferedReader;
//import java.io.DataInputStream;
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.FileOutputStream;
//import java.io.IOException;
//import java.io.InputStream;
//import java.io.InputStreamReader;
//import java.io.OutputStream;
//import java.nio.ByteBuffer;
//import java.nio.ByteOrder;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//
//public class MainActivity extends AppCompatActivity {
//    private static final String TAG = "SpeechRecognition";
//    private static final int PERMISSION_REQUEST_CODE = 200;
//    private static final int SAMPLE_RATE = 16000;
//    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
//    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
//    private static final int BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
//
//    private Button btnRecordMic;
//    private Button btnLoadFile;
//    private Button btnStopRecording;
//    private TextView tvResult;
//    private TextView tvStatus;
//
//    private AudioRecord audioRecord;
//    private boolean isRecording = false;
//    private Module speechModel = null;
//    private List<Short> recordedAudio = new ArrayList<>();
//    private Map<Integer, String> vocabulary = new HashMap<>();
//    private int blankIndex = 0;
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//
//        btnRecordMic = findViewById(R.id.btnRecordMic);
//        btnLoadFile = findViewById(R.id.btnLoadFile);
//        btnStopRecording = findViewById(R.id.btnStopRecording);
//        tvResult = findViewById(R.id.tvResult);
//        tvStatus = findViewById(R.id.tvStatus);
//
//        btnStopRecording.setEnabled(false);
//
//        if (!checkPermissions()) {
//            requestPermissions();
//        }
//        new Thread(() -> {
//            try {
//                long startTime = System.currentTimeMillis();
//                speechModel = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "chunkformer_quantized_4.ptl"));
//                long loadTime = System.currentTimeMillis() - startTime;
//                runOnUiThread(() -> {
//                    tvStatus.setText("Model loaded in " + (loadTime / 1000.0) + "s");
//                    Log.d(TAG, "✅ Model loaded successfully in " + loadTime + "ms");
//                });
//            } catch (IOException e) {
//                Log.e(TAG, "Error loading model", e);
//                runOnUiThread(() -> {
//                    tvStatus.setText("Error loading model");
//                    Toast.makeText(MainActivity.this, "Failed to load model", Toast.LENGTH_SHORT).show();
//                });
//            }
//        }).start();
//
//        try {
//            loadVocabulary();
//            Log.d(TAG, "✅ Vocabulary loaded: " + vocabulary.size() + " tokens, blankIndex=" + blankIndex);
//        } catch (IOException e) {
//            Log.e(TAG, "Error loading vocabulary", e);
//            tvStatus.setText("Error loading vocabulary");
//            Toast.makeText(this, "Failed to load vocabulary", Toast.LENGTH_SHORT).show();
//        }
//
//        btnRecordMic.setOnClickListener(v -> {
//            if (checkPermissions()) {
//                startRecording();
//            } else {
//                requestPermissions();
//            }
//        });
//
//        btnStopRecording.setOnClickListener(v -> stopRecording());
//
//        btnLoadFile.setOnClickListener(v -> {
//            try {
//                processAudioFile("studentbook1.wav");
//            } catch (IOException e) {
//                Log.e(TAG, "Error processing audio file", e);
//                Toast.makeText(this, "Error processing audio file", Toast.LENGTH_SHORT).show();
//            }
//        });
//    }
//
//    private void processAudioFile(String fileName) throws IOException {
//        tvStatus.setText("Loading audio file...");
//        tvResult.setText("");
//
//        new Thread(() -> {
//            try {
//                String filePath = assetFilePath(getApplicationContext(), fileName);
//                short[] audioData = loadWavFile(filePath);
//
//                Log.d(TAG, "🔊 Loaded audio: " + audioData.length + " samples");
//
//                String transcription = recognizeSpeechChunked(audioData);
//
//                runOnUiThread(() -> {
//                    tvResult.setText(transcription);
//                    tvStatus.setText("Ready");
//                });
//            } catch (IOException e) {
//                Log.e(TAG, "Error loading audio file", e);
//                runOnUiThread(() -> {
//                    tvStatus.setText("Error loading file");
//                    Toast.makeText(MainActivity.this, "Error loading audio file", Toast.LENGTH_SHORT).show();
//                });
//            }
//        }).start();
//    }
//
//    private String recognizeSpeechChunked(short[] audioData) {
//        if (speechModel == null) {
//            return "Model not loaded";
//        }
//
//        long totalStartTime = System.currentTimeMillis();
//        float audioDuration = audioData.length / (float) SAMPLE_RATE;
//        Log.d(TAG, "📊 Audio duration: " + String.format("%.2f", audioDuration) + "s (" + audioData.length + " samples)");
//
//        String result = recognizeSpeech(audioData);
//
//        long totalTime = System.currentTimeMillis() - totalStartTime;
//        Log.d(TAG, "⏱️ Total processing time: " + (totalTime / 1000.0) + "s");
//
//        return result;
//    }
//
//    private String recognizeSpeech(short[] audioData) {
//        if (speechModel == null) return "Model not loaded";
//
//        try {
//            long t1 = System.currentTimeMillis();
//            float[][] features = extractFbankFeatures(audioData);
//            if (features == null || features.length == 0) {
//                Log.e(TAG, "Feature extraction failed or produced no features.");
//                return "[FEATURE EXTRACTION FAILED]";
//            }
//            long t2 = System.currentTimeMillis();
//            Log.d(TAG, "⏱️ Feature extraction (Java Impl): " + (t2 - t1) + "ms");
//
//            float[] flatFeatures = flattenArray(features);
//            int timeSteps = features.length;
//            int featureDim = features[0].length;
//
//            Tensor inputTensor = Tensor.fromBlob(flatFeatures, new long[]{1, timeSteps, featureDim});
//            Tensor lensTensor = Tensor.fromBlob(new long[]{timeSteps}, new long[]{1});
//
//            long t5 = System.currentTimeMillis();
//            IValue output = speechModel.forward(IValue.from(inputTensor), IValue.from(lensTensor));
//            long t6 = System.currentTimeMillis();
//            Log.d(TAG, "⏱️ MODEL INFERENCE: " + (t6 - t5) + "ms");
//
//            IValue[] elements = output.toTuple();
//            Tensor outputTensor = elements[0].toTensor();
//
//            return decodeCtcOutput(outputTensor.getDataAsFloatArray(), outputTensor.shape());
//
//        } catch (Exception e) {
//            Log.e(TAG, "Error during inference", e);
//            e.printStackTrace();
//            return "Error: " + e.getMessage();
//        }
//    }
//
//    private float[][] extractFbankFeatures(short[] audioData) {
//        float sampleRate = 16000.0f;
//        int numMelBins = 80;
//        float frameLengthMs = 25.0f;
//        float frameShiftMs = 10.0f;
//        float dither = 0.0f;
//
//        // 🔥 DEBUG: In ra thông số tính toán
//        int frameLength = (int) (frameLengthMs * sampleRate / 1000.0f);  // = 400
//        int frameShift = (int) (frameShiftMs * sampleRate / 1000.0f);    // = 160
//
//        Log.d(TAG, String.format("🔧 Config: frameLength=%d, frameShift=%d, audioSamples=%d",
//                frameLength, frameShift, audioData.length));
//
//        // 1️⃣ Convert to float waveform
//        float[] waveform = new float[audioData.length];
//        for (int i = 0; i < audioData.length; i++) {
//            waveform[i] = audioData[i];
//        }
//
//        // 2️⃣ Apply dither (nếu cần, hiện tại = 0)
//        if (dither != 0.0f) {
//            for (int i = 0; i < waveform.length; i++) {
//                waveform[i] += (float) (Math.random() * 2 - 1) * dither;
//            }
//        }
//
//        if (waveform.length < frameLength) {
//            Log.e(TAG, "❌ Audio too short!");
//            return null;
//        }
//
//        // ✅ CRITICAL: Công thức chính xác từ Kaldi
//        // numFrames = floor((waveform.length - frameLength) / frameShift) + 1
//        int numFrames = (waveform.length - frameLength) / frameShift + 1;
//
//        Log.d(TAG, String.format("📊 Expected frames: (%d - %d) / %d + 1 = %d",
//                waveform.length, frameLength, frameShift, numFrames));
//
//        // 3️⃣ Create Povey window
//        float[] window = new float[frameLength];
//        for (int i = 0; i < frameLength; i++) {
//            float val = 0.5f - 0.5f * (float) Math.cos(2.0 * Math.PI * i / (frameLength - 1));
//            window[i] = (float) Math.pow(val, 0.85);
//        }
//
//        int fftSize = 1;
//        while (fftSize < frameLength) {
//            fftSize *= 2;
//        }
//        int numFreqs = fftSize / 2 + 1;
//        float[][] powerSpectrum = new float[numFrames][numFreqs];
//        FloatFFT_1D fft = new FloatFFT_1D(fftSize);
//
//        for (int i = 0; i < numFrames; i++) {
//            float[] frame = new float[fftSize];
//            int start = i * frameShift;
//
//            // ✅ SỬA 2: Remove DC offset TRƯỚC KHI nhân window
//            float mean = 0.0f;
//            for (int j = 0; j < frameLength; j++) {
//                mean += waveform[start + j];
//            }
//            mean /= frameLength;
//
//            for (int j = 0; j < frameLength; j++) {
//                frame[j] = (waveform[start + j] - mean) * window[j];
//            }
//
//            // 4️⃣ FFT
//            fft.realForward(frame);
//
//            // ✅ SỬA 3: Tính power spectrum ĐÚNG
//            powerSpectrum[i][0] = frame[0] * frame[0];
//            for (int k = 1; k < fftSize / 2; k++) {
//                float real = frame[2 * k];
//                float imag = frame[2 * k + 1];
//                powerSpectrum[i][k] = real * real + imag * imag;
//            }
//            // Nyquist frequency bin (nếu fftSize chẵn)
//            powerSpectrum[i][fftSize / 2] = frame[1] * frame[1];
//        }
//
//        // 5️⃣ Apply Mel filterbank
//        float[][] melFilters = createMelFilterbank(sampleRate, fftSize, numMelBins);
//
//        float[][] fbankFeatures = new float[numFrames][numMelBins];
//        for (int i = 0; i < numFrames; i++) {
//            for (int m = 0; m < numMelBins; m++) {
//                float melEnergy = 0.0f;
//                for (int k = 0; k < numFreqs; k++) {
//                    melEnergy += powerSpectrum[i][k] * melFilters[m][k];
//                }
//                // ✅ Kaldi không dùng energy_floor ở đây, chỉ clamp nếu <= 0
//                if (melEnergy <= 0.0f) {
//                    melEnergy = 1e-10f; // Tránh log(0)
//                }
//                fbankFeatures[i][m] = (float) Math.log(melEnergy);
//            }
//        }
//
//        // 🔍 DEBUG: In chi tiết features để so sánh với Python
//        logFeatureStats(fbankFeatures);
//
//        // In thêm 3 frames đầu để debug
//        Log.d(TAG, "Frame 0: " + Arrays.toString(Arrays.copyOfRange(fbankFeatures[0], 0, 10)));
//        if (numFrames > 1) {
//            Log.d(TAG, "Frame 1: " + Arrays.toString(Arrays.copyOfRange(fbankFeatures[1], 0, 10)));
//        }
//        if (numFrames > 2) {
//            Log.d(TAG, "Frame 2: " + Arrays.toString(Arrays.copyOfRange(fbankFeatures[2], 0, 10)));
//        }
//
//        return fbankFeatures;
//    }
//
//    private float[][] createMelFilterbank(float sampleRate, int fftSize, int numMelBins) {
//        int numFreqs = fftSize / 2 + 1;
//        float lowFreq = 0.0f;
//        float highFreq = sampleRate / 2.0f;
//
//        float lowMel = hzToMel(lowFreq);
//        float highMel = hzToMel(highFreq);
//
//        float[] melPoints = new float[numMelBins + 2];
//        float melSpacing = (highMel - lowMel) / (numMelBins + 1);
//        for (int i = 0; i < melPoints.length; i++) {
//            melPoints[i] = lowMel + i * melSpacing;
//        }
//
//        float[] hzPoints = new float[melPoints.length];
//        for (int i = 0; i < melPoints.length; i++) {
//            hzPoints[i] = melToHz(melPoints[i]);
//        }
//
//        float[] binPoints = new float[hzPoints.length];
//        float fftBinWidth = sampleRate / fftSize;
//        for (int i = 0; i < hzPoints.length; i++) {
//            binPoints[i] = hzPoints[i] / fftBinWidth;
//        }
//
//        float[][] filters = new float[numMelBins][numFreqs];
//        for (int m = 0; m < numMelBins; m++) {
//            float leftMelBin = binPoints[m];
//            float centerMelBin = binPoints[m + 1];
//            float rightMelBin = binPoints[m + 2];
//
//            for (int k = 0; k < numFreqs; k++) {
//                float val = 0.0f;
//                if (k >= leftMelBin && k <= centerMelBin) {
//                    if (centerMelBin > leftMelBin) {
//                        val = (k - leftMelBin) / (centerMelBin - leftMelBin);
//                    }
//                } else if (k > centerMelBin && k <= rightMelBin) {
//                    if (rightMelBin > centerMelBin) {
//                        val = (rightMelBin - k) / (rightMelBin - centerMelBin);
//                    }
//                }
//                filters[m][k] = val;
//            }
//        }
//        return filters;
//    }
//
//    private float hzToMel(float hz) {
//        return 2595.0f * (float) Math.log10(1.0f + hz / 700.0f);
//    }
//
//    private float melToHz(float mel) {
//        return 700.0f * ((float) Math.pow(10.0f, mel / 2595.0f) - 1.0f);
//    }
//
//    private void logFeatureStats(float[][] features) {
//        if (features == null || features.length == 0) {
//            Log.e(TAG, "❌ Features are null or empty!");
//            return;
//        }
//        float sum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
//        for (float[] feature : features) {
//            for (float v : feature) {
//                sum += v;
//                min = Math.min(min, v);
//                max = Math.max(max, v);
//            }
//        }
//        float mean = sum / (features.length * features[0].length);
//        Log.d(TAG, String.format("✅ Features (Java Impl): shape=[%d, %d], mean=%.2f, min=%.2f, max=%.2f",
//                features.length, features[0].length, mean, min, max));
//        Log.d(TAG, "✅ First frame (Java Impl): " + Arrays.toString(Arrays.copyOfRange(features[0], 0, 5)));
//    }
//
//    private String decodeCtcOutput(float[] output, long[] shape) {
//        int timeSteps = (int) shape[1];
//        int vocabSize = (int) shape[2];
//
//        StringBuilder transcription = new StringBuilder();
//        int prevToken = -1;
//
//        for (int t = 0; t < timeSteps; t++) {
//            int maxIdx = 0;
//            float maxVal = output[t * vocabSize];
//            for (int v = 1; v < vocabSize; v++) {
//                float val = output[t * vocabSize + v];
//                if (val > maxVal) {
//                    maxVal = val;
//                    maxIdx = v;
//                }
//            }
//
//            if (maxIdx != blankIndex && maxIdx != prevToken) {
//                String token = vocabulary.get(maxIdx);
//                if (token != null && !"<unk>".equals(token)) {
//                    transcription.append(token);
//                }
//            }
//            prevToken = maxIdx;
//        }
//
//        String rawResult = transcription.toString();
////        String finalResult = rawResult.replace(" ", " ").trim().replaceAll(" +", " ");
//        String finalResult = rawResult
//                .replace("▁", " ")
//                .replace("_", " ")
//                .trim()
//                .replaceAll(" +", " ");
//
//        finalResult = finalResult.replaceAll("[:.]+$", "");
//
//        if (finalResult.isEmpty()) {
//            Log.w(TAG, "⚠️ Empty transcript! Total frames: " + timeSteps);
//            return "[NO SPEECH DETECTED]";
//        }
//        return finalResult;
//    }
//
//    private void loadVocabulary() throws IOException {
//        vocabulary.clear();
//        InputStream is = getAssets().open("vocab.txt");
//        BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
//        String line;
//        while ((line = reader.readLine()) != null) {
//            line = line.trim();
//            if (line.isEmpty()) continue;
//            String[] parts = line.split("\\s+", 2);
//            if (parts.length >= 2) {
//                try {
//                    String token = parts[0];
//                    int id = Integer.parseInt(parts[1].trim());
//                    vocabulary.put(id, token);
//                    if ("<blank>".equals(token)) {
//                        blankIndex = id;
//                    }
//                } catch (NumberFormatException e) {
//                    Log.w(TAG, "Failed to parse vocab line: " + line);
//                }
//            }
//        }
//        reader.close();
//        is.close();
//        if (blankIndex == 0 && !vocabulary.containsValue("<blank>")) {
//            Log.w(TAG, "⚠️ <blank> not found in vocab — using index 0 as blank");
//        }
//    }
//
//    private void startRecording() {
//        if (audioRecord != null) {
//            audioRecord.release();
//        }
//        recordedAudio.clear();
//        isRecording = true;
//        try {
//            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
//                return;
//            }
//            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, BUFFER_SIZE);
//            audioRecord.startRecording();
//            btnRecordMic.setEnabled(false);
//            btnLoadFile.setEnabled(false);
//            btnStopRecording.setEnabled(true);
//            tvStatus.setText("Recording... Speak now!");
//            new Thread(() -> {
//                short[] buffer = new short[BUFFER_SIZE / 2];
//                while (isRecording) {
//                    int read = audioRecord.read(buffer, 0, buffer.length);
//                    if (read > 0) {
//                        for (int i = 0; i < read; i++) {
//                            recordedAudio.add(buffer[i]);
//                        }
//                    }
//                }
//            }).start();
//        } catch (SecurityException e) {
//            Log.e(TAG, "Permission denied", e);
//            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_SHORT).show();
//        }
//    }
//
//    private void stopRecording() {
//        isRecording = false;
//        if (audioRecord != null) {
//            try {
//                audioRecord.stop();
//                audioRecord.release();
//            } catch (IllegalStateException e) {
//                Log.e(TAG, "AudioRecord stop failed", e);
//            }
//            audioRecord = null;
//        }
//        btnRecordMic.setEnabled(true);
//        btnLoadFile.setEnabled(true);
//        btnStopRecording.setEnabled(false);
//        tvStatus.setText("Processing audio...");
//        new Thread(() -> {
//            short[] audioData = new short[recordedAudio.size()];
//            for (int i = 0; i < recordedAudio.size(); i++) {
//                audioData[i] = recordedAudio.get(i);
//            }
//            String transcription = recognizeSpeechChunked(audioData);
//            runOnUiThread(() -> {
//                tvResult.setText(transcription);
//                tvStatus.setText("Ready");
//            });
//        }).start();
//    }
//
//    private short[] loadWavFile(String filePath) throws IOException {
//        FileInputStream fis = new FileInputStream(filePath);
//        DataInputStream dis = new DataInputStream(new java.io.BufferedInputStream(fis));
//
//        try {
//            // === ĐỌC VÀ VERIFY WAV HEADER ===
//
//            // 1. RIFF header
//            byte[] riffHeader = new byte[4];
//            dis.readFully(riffHeader);
//            String riff = new String(riffHeader);
//            if (!riff.equals("RIFF")) {
//                throw new IOException("Not a valid WAV file (missing RIFF)");
//            }
//
//            // 2. File size (skip)
//            dis.skipBytes(4);
//
//            // 3. WAVE header
//            byte[] waveHeader = new byte[4];
//            dis.readFully(waveHeader);
//            String wave = new String(waveHeader);
//            if (!wave.equals("WAVE")) {
//                throw new IOException("Not a valid WAV file (missing WAVE)");
//            }
//
//            // 4. Find "fmt " chunk
//            byte[] fmtHeader = new byte[4];
//            dis.readFully(fmtHeader);
//            String fmt = new String(fmtHeader);
//            if (!fmt.equals("fmt ")) {
//                throw new IOException("Not a valid WAV file (missing fmt)");
//            }
//
//            // 5. fmt chunk size
//            byte[] fmtSizeBytes = new byte[4];
//            dis.readFully(fmtSizeBytes);
//            int fmtSize = ByteBuffer.wrap(fmtSizeBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
//
//            // 6. Audio format (1 = PCM)
//            byte[] audioFormatBytes = new byte[2];
//            dis.readFully(audioFormatBytes);
//            int audioFormat = ByteBuffer.wrap(audioFormatBytes).order(ByteOrder.LITTLE_ENDIAN).getShort();
//
//            // 7. Number of channels
//            byte[] numChannelsBytes = new byte[2];
//            dis.readFully(numChannelsBytes);
//            int numChannels = ByteBuffer.wrap(numChannelsBytes).order(ByteOrder.LITTLE_ENDIAN).getShort();
//
//            // 8. Sample rate
//            byte[] sampleRateBytes = new byte[4];
//            dis.readFully(sampleRateBytes);
//            int sampleRate = ByteBuffer.wrap(sampleRateBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
//
//            // 9. Byte rate (skip)
//            dis.skipBytes(4);
//
//            // 10. Block align (skip)
//            dis.skipBytes(2);
//
//            // 11. Bits per sample
//            byte[] bitsPerSampleBytes = new byte[2];
//            dis.readFully(bitsPerSampleBytes);
//            int bitsPerSample = ByteBuffer.wrap(bitsPerSampleBytes).order(ByteOrder.LITTLE_ENDIAN).getShort();
//
//            // 🔍 LOG WAV INFO
//            Log.d(TAG, String.format("📂 WAV Info: format=%d, channels=%d, sampleRate=%d, bits=%d",
//                    audioFormat, numChannels, sampleRate, bitsPerSample));
//
//            // ✅ VALIDATE FORMAT
//            if (audioFormat != 1) {
//                throw new IOException("Only PCM format supported (got format " + audioFormat + ")");
//            }
//            if (numChannels > 2) {
//                throw new IOException("Only mono/stereo audio supported (got " + numChannels + " channels)");
//            }
//            if (bitsPerSample != 16) {
//                throw new IOException("Only 16-bit audio supported (got " + bitsPerSample + " bits)");
//            }
//
//            // Log warnings cho format cần convert
//            if (numChannels == 2) {
//                Log.w(TAG, "⚠️ Stereo audio detected - will convert to mono");
//            }
//            if (sampleRate != 16000) {
//                Log.w(TAG, "⚠️ Sample rate " + sampleRate + "Hz - will resample to 16kHz");
//            }
//
//            // Skip extra fmt bytes (if any)
//            if (fmtSize > 16) {
//                dis.skipBytes(fmtSize - 16);
//            }
//
//            // 12. Find "data" chunk
//            byte[] dataHeader = new byte[4];
//            dis.readFully(dataHeader);
//            String data = new String(dataHeader);
//
//            // Skip other chunks until we find "data"
//            while (!data.equals("data")) {
//                byte[] chunkSizeBytes = new byte[4];
//                dis.readFully(chunkSizeBytes);
//                int chunkSize = ByteBuffer.wrap(chunkSizeBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
//                dis.skipBytes(chunkSize); // Skip this chunk
//                dis.readFully(dataHeader);
//                data = new String(dataHeader);
//            }
//
//            // 13. Data chunk size
//            byte[] dataSizeBytes = new byte[4];
//            dis.readFully(dataSizeBytes);
//            int dataSize = ByteBuffer.wrap(dataSizeBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();
//
//            Log.d(TAG, "📊 Data size: " + dataSize + " bytes (" + (dataSize / 2) + " samples)");
//
//            // 14. Read audio data
//            byte[] audioBytes = new byte[dataSize];
//            dis.readFully(audioBytes);
//
//            // 15. Convert to short array
//            short[] audioData = new short[audioBytes.length / 2];
//            ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(audioData);
//
//            Log.d(TAG, "✅ Loaded " + audioData.length + " samples");
//
//            // ✅ CRITICAL: Xử lý STEREO TRƯỚC, RESAMPLE SAU
//            // (vì stereo có interleaved L/R channels, phải tách trước khi resample)
//            if (numChannels == 2) {
//                Log.w(TAG, "⚠️ Converting stereo to mono...");
//                audioData = stereoToMono(audioData);
//            }
//
//            if (sampleRate != 16000) {
//                Log.w(TAG, "⚠️ Resampling from " + sampleRate + "Hz to 16000Hz...");
//                audioData = resampleAudio(audioData, sampleRate, 16000);
//            }
//
//            return audioData;
//
//        } finally {
//            dis.close();
//        }
//    }
//
//    private short[] resampleAudio(short[] input, int inputRate, int outputRate) {
//        double ratio = (double) inputRate / outputRate;
//        int outputLength = (int) (input.length / ratio);
//        short[] output = new short[outputLength];
//
//        for (int i = 0; i < outputLength; i++) {
//            double inputIndex = i * ratio;
//            int index1 = (int) inputIndex;
//            int index2 = Math.min(index1 + 1, input.length - 1);
//            double fraction = inputIndex - index1;
//
//            // Linear interpolation
//            output[i] = (short) ((1 - fraction) * input[index1] + fraction * input[index2]);
//        }
//
//        Log.d(TAG, "✅ Resampled: " + input.length + " → " + output.length + " samples");
//        return output;
//    }
//
//    private short[] stereoToMono(short[] stereo) {
//        short[] mono = new short[stereo.length / 2];
//        for (int i = 0; i < mono.length; i++) {
//            int left = stereo[i * 2];
//            int right = stereo[i * 2 + 1];
//            mono[i] = (short) ((left + right) / 2);
//        }
//        Log.d(TAG, "✅ Converted stereo to mono: " + stereo.length + " → " + mono.length + " samples");
//        return mono;
//    }
//    private float[] flattenArray(float[][] array) {
//        if (array == null || array.length == 0) return new float[0];
//        int rows = array.length;
//        int cols = array[0].length;
//        float[] flat = new float[rows * cols];
//        for (int i = 0; i < rows; i++) {
//            System.arraycopy(array[i], 0, flat, i * cols, cols);
//        }
//        return flat;
//    }
//
//    public static String assetFilePath(Context context, String assetName) throws IOException {
//        File file = new File(context.getFilesDir(), assetName);
//        if (file.exists() && file.length() > 0) {
//            return file.getAbsolutePath();
//        }
//        try (InputStream is = context.getAssets().open(assetName);
//             OutputStream os = new FileOutputStream(file)) {
//            byte[] buffer = new byte[4096];
//            int read;
//            while ((read = is.read(buffer)) != -1) {
//                os.write(buffer, 0, read);
//            }
//        }
//        return file.getAbsolutePath();
//    }
//
//    private boolean checkPermissions() {
//        int recordPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO);
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
//            return recordPermission == PackageManager.PERMISSION_GRANTED;
//        } else {
//            int readPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
//            return recordPermission == PackageManager.PERMISSION_GRANTED && readPermission == PackageManager.PERMISSION_GRANTED;
//        }
//    }
//
//    private void requestPermissions() {
//        String[] permissions;
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
//            permissions = new String[]{Manifest.permission.RECORD_AUDIO};
//        } else {
//            permissions = new String[]{Manifest.permission.RECORD_AUDIO, Manifest.permission.READ_EXTERNAL_STORAGE};
//        }
//        ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
//    }
//
//    @Override
//    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
//        if (requestCode == PERMISSION_REQUEST_CODE) {
//            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
//            } else {
//                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
//            }
//        }
//    }
//
//    @Override
//    protected void onDestroy() {
//        super.onDestroy();
//        if (audioRecord != null) {
//            audioRecord.release();
//        }
//    }
//}
