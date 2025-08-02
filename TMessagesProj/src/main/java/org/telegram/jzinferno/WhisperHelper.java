package org.telegram.jzinferno;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.util.Log;

import org.telegram.messenger.ApplicationLoader;
import org.telegram.messenger.FileLog;
import org.telegram.messenger.LocaleController;
import org.telegram.messenger.R;
import org.telegram.messenger.UserConfig;
import org.telegram.ui.Components.AlertsCreator;
import org.telegram.ui.Components.BulletinFactory;
import org.telegram.ui.LaunchActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiConsumer;

import com.jzinferno.whisper.LibWhisper;

public class WhisperHelper {
    private static final String TAG = "WhisperHelper";
    private static final String MODEL_FILENAME = "ggml-tiny-q8_0.bin";
    private static final String WHISPER_MODELS_DIR = "whisper/models";
    private static final String WHISPER_AUDIO_DIR = "whisper/audio";

    private static final ExecutorService executor = Executors.newCachedThreadPool();
    private static volatile String modelPath;
    private static final Random random = new Random();

    public static boolean useLocalTranscribe(int account) {
        return !UserConfig.getInstance(account).isPremium();
    }

    public static void showErrorDialog(Exception e) {
        var fragment = LaunchActivity.getSafeLastFragment();
        var message = e.getLocalizedMessage();

        if (!BulletinFactory.canShowBulletin(fragment) || message == null) return;

        if (message.length() > 45) {
            AlertsCreator.showSimpleAlert(fragment,
                    LocaleController.getString(R.string.ErrorOccurred), e.getMessage());
        } else {
            BulletinFactory.of(fragment).createErrorBulletin(message).show();
        }
    }

    public static void requestLocalTranscribe(String path, boolean isVideo, BiConsumer<String, Exception> callback) {
        executor.submit(() -> processTranscription(path, isVideo, callback));
    }

    private static void processTranscription(String path, boolean isVideo, BiConsumer<String, Exception> callback) {
        String modelPath = getModelPath();
        if (modelPath == null) {
            callback.accept(null, new RuntimeException("Error model path is not set"));
            return;
        }

        List<File> tempFiles = new ArrayList<>();
        try {
            String wavPath = convertToWav(path, tempFiles);
            String result = LibWhisper.transcribe(modelPath, wavPath, "auto", 4);
            callback.accept(result.trim(), null);
        } catch (Exception e) {
            FileLog.e("Whisper transcription failed", e);
            callback.accept(null, e);
        } finally {
            cleanupFiles(tempFiles);
        }
    }

    private static String getModelPath() {
        if (modelPath != null && new File(modelPath).exists()) {
            return modelPath;
        }

        try {
            File modelsDir = new File(ApplicationLoader.applicationContext.getFilesDir(), WHISPER_MODELS_DIR);
            if (!modelsDir.exists() && !modelsDir.mkdirs()) {
                throw new IOException("Failed to create models directory: " + modelsDir.getAbsolutePath());
            }

            File modelFile = new File(modelsDir, MODEL_FILENAME);

            if (!modelFile.exists()) {
                extractModelFromAssets(modelFile);
            }

            modelPath = modelFile.getAbsolutePath();
            return modelPath;

        } catch (IOException e) {
            FileLog.e("Failed to setup Whisper model", e);
            return null;
        }
    }

    private static void extractModelFromAssets(File modelFile) throws IOException {
        Context context = ApplicationLoader.applicationContext;

        try (InputStream in = context.getAssets().open(MODEL_FILENAME);
             FileOutputStream out = new FileOutputStream(modelFile)) {

            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        }
    }

    private static String convertToWav(String inputPath, List<File> tempFiles) throws IOException {
        String lowerPath = inputPath.toLowerCase();

        if (lowerPath.endsWith(".wav")) {
            return inputPath;
        }

        File audioDir = new File(ApplicationLoader.applicationContext.getFilesDir(), WHISPER_AUDIO_DIR);
        if (!audioDir.exists() && !audioDir.mkdirs()) {
            throw new IOException("Failed to create audio directory: " + audioDir.getAbsolutePath());
        }

        String randomName = generateRandomName();
        File wavFile = new File(audioDir, randomName + ".wav");
        tempFiles.add(wavFile);

        convertToWavFile(inputPath, wavFile);

        if (!wavFile.exists() || wavFile.length() == 0) {
            throw new IOException("Audio conversion failed or produced empty file");
        }

        return wavFile.getAbsolutePath();
    }

    private static void convertToWavFile(String inputPath, File outputFile) throws IOException {
        MediaExtractor extractor = null;
        MediaCodec codec = null;
        FileOutputStream fos = null;

        try {
            extractor = new MediaExtractor();
            extractor.setDataSource(inputPath);

            MediaFormat audioFormat = null;
            int audioTrackIndex = -1;
            for (int i = 0; i < extractor.getTrackCount(); i++) {
                MediaFormat format = extractor.getTrackFormat(i);
                String mime = format.getString(MediaFormat.KEY_MIME);
                if (mime != null && mime.startsWith("audio/")) {
                    audioFormat = format;
                    audioTrackIndex = i;
                    break;
                }
            }

            if (audioFormat == null) {
                throw new IOException("No audio track found");
            }

            String mime = audioFormat.getString(MediaFormat.KEY_MIME);
            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(audioFormat, null, null, 0);
            codec.start();

            extractor.selectTrack(audioTrackIndex);

            fos = new FileOutputStream(outputFile);

            writeWavHeader(fos, 0);

            ByteBuffer[] inputBuffers = codec.getInputBuffers();
            ByteBuffer[] outputBuffers = codec.getOutputBuffers();
            MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();

            boolean isEOS = false;
            long totalPcmBytes = 0;

            while (!isEOS) {
                int inputBufferIndex = codec.dequeueInputBuffer(10000);
                if (inputBufferIndex >= 0) {
                    ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
                    int sampleSize = extractor.readSampleData(inputBuffer, 0);

                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        isEOS = true;
                    } else {
                        codec.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.getSampleTime(), 0);
                        extractor.advance();
                    }
                }

                int outputBufferIndex = codec.dequeueOutputBuffer(bufferInfo, 10000);
                if (outputBufferIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                    outputBuffers = codec.getOutputBuffers();
                } else if (outputBufferIndex >= 0) {
                    ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];

                    if (bufferInfo.size > 0) {
                        byte[] pcmData = convertPcmToWhisperFormat(outputBuffer, bufferInfo, audioFormat);
                        if (pcmData != null && pcmData.length > 0) {
                            fos.write(pcmData);
                            totalPcmBytes += pcmData.length;
                        }
                    }

                    codec.releaseOutputBuffer(outputBufferIndex, false);

                    if ((bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break;
                    }
                }
            }

            fos.close();
            fos = null;
            updateWavHeader(outputFile, totalPcmBytes);

        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    Log.w(TAG, "Error closing FileOutputStream", e);
                }
            }
            if (codec != null) {
                try {
                    codec.stop();
                    codec.release();
                } catch (Exception e) {
                    Log.w(TAG, "Error releasing codec", e);
                }
            }
            if (extractor != null) {
                try {
                    extractor.release();
                } catch (Exception e) {
                    Log.w(TAG, "Error releasing extractor", e);
                }
            }
        }
    }

    private static byte[] convertPcmToWhisperFormat(ByteBuffer outputBuffer, MediaCodec.BufferInfo bufferInfo, MediaFormat originalFormat) {
        outputBuffer.position(bufferInfo.offset);
        outputBuffer.limit(bufferInfo.offset + bufferInfo.size);

        int originalSampleRate = originalFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
        int channelCount = originalFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);

        int sampleCount = bufferInfo.size / 2;
        short[] samples = new short[sampleCount];

        outputBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < sampleCount; i++) {
            samples[i] = outputBuffer.getShort();
        }

        if (channelCount > 1) {
            samples = convertToMono(samples, channelCount);
        }

        if (originalSampleRate != 16000) {
            samples = resample(samples, originalSampleRate, 16000);
        }

        ByteBuffer result = ByteBuffer.allocate(samples.length * 2);
        result.order(ByteOrder.LITTLE_ENDIAN);
        for (short sample : samples) {
            result.putShort(sample);
        }
        return result.array();
    }

    private static short[] convertToMono(short[] stereoSamples, int channelCount) {
        int monoLength = stereoSamples.length / channelCount;
        short[] monoSamples = new short[monoLength];

        for (int i = 0; i < monoLength; i++) {
            if (channelCount == 2) {
                long sum = (long) stereoSamples[i * 2] + stereoSamples[i * 2 + 1];
                monoSamples[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, sum));
            } else {
                long sum = 0;
                for (int ch = 0; ch < channelCount; ch++) {
                    sum += stereoSamples[i * channelCount + ch];
                }
                monoSamples[i] = (short) (sum / channelCount);
            }
        }

        return monoSamples;
    }

    private static short[] resample(short[] input, int inputRate, int outputRate) {
        if (inputRate == outputRate) {
            return input;
        }

        double ratio = (double) inputRate / outputRate;
        int outputLength = (int) (input.length / ratio);
        short[] output = new short[outputLength];

        for (int i = 0; i < outputLength; i++) {
            double srcIndex = i * ratio;
            int srcIndexInt = (int) srcIndex;

            if (srcIndexInt >= input.length - 1) {
                output[i] = input[input.length - 1];
            } else {
                double fraction = srcIndex - srcIndexInt;
                double sample1 = input[srcIndexInt];
                double sample2 = input[srcIndexInt + 1];
                output[i] = (short) (sample1 + fraction * (sample2 - sample1));
            }
        }

        return output;
    }

    private static void writeWavHeader(FileOutputStream fos, long dataSize) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(44);
        header.order(ByteOrder.LITTLE_ENDIAN);
        header.put("RIFF".getBytes());
        header.putInt((int) (dataSize + 36));
        header.put("WAVE".getBytes());
        header.put("fmt ".getBytes());
        header.putInt(16);
        header.putShort((short) 1);
        header.putShort((short) 1);
        header.putInt(16000);
        header.putInt(32000);
        header.putShort((short) 2);
        header.putShort((short) 16);
        header.put("data".getBytes());
        header.putInt((int) dataSize);
        fos.write(header.array());
    }

    private static void updateWavHeader(File wavFile, long totalPcmBytes) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(wavFile, "rw")) {
            long totalDataLen = totalPcmBytes + 36;
            raf.seek(4);
            raf.write((int) (totalDataLen & 0xff));
            raf.write((int) ((totalDataLen >> 8) & 0xff));
            raf.write((int) ((totalDataLen >> 16) & 0xff));
            raf.write((int) ((totalDataLen >> 24) & 0xff));
            raf.seek(40);
            raf.write((int) (totalPcmBytes & 0xff));
            raf.write((int) ((totalPcmBytes >> 8) & 0xff));
            raf.write((int) ((totalPcmBytes >> 16) & 0xff));
            raf.write((int) ((totalPcmBytes >> 24) & 0xff));
        }
    }

    private static String generateRandomName() {
        StringBuilder sb = new StringBuilder(10);
        for (int i = 0; i < 10; i++) {
            char c = (char) ('a' + random.nextInt(26));
            sb.append(c);
        }

        return sb.toString();
    }

    private static void cleanupFiles(List<File> files) {
        for (File file : files) {
            if (file != null && file.exists()) {
                try {
                    if (!file.delete()) {
                        Log.w(TAG, "Failed to delete temp file: " + file.getAbsolutePath());
                    }
                } catch (Exception e) {
                    Log.w(TAG, "Error deleting temp file", e);
                }
            }
        }
    }
}
