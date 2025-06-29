package org.telegram.jzinferno;

import android.content.Context;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.util.Log;

import org.telegram.messenger.ApplicationLoader;
import org.telegram.messenger.FileLog;
import org.telegram.messenger.LocaleController;
import org.telegram.messenger.R;
import org.telegram.messenger.UserConfig;
import org.telegram.ui.Components.AlertsCreator;
import org.telegram.ui.Components.BulletinFactory;
import org.telegram.ui.LaunchActivity;

import com.arthenica.ffmpegkit.FFmpegKit;
import com.jzinferno.whisper.LibWhisper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiConsumer;

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
            String audioPath = isVideo ? extractAudioFromVideo(path, tempFiles) : path;
            String wavPath = convertToWavWithFFmpeg(audioPath, tempFiles);
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

    private static String extractAudioFromVideo(String videoPath, List<File> tempFiles) throws IOException {
        File audioFile = new File(videoPath + ".m4a");
        tempFiles.add(audioFile);

        MediaExtractor extractor = null;
        MediaMuxer muxer = null;
        boolean muxerStarted = false;

        try {
            extractor = new MediaExtractor();
            extractor.setDataSource(videoPath);

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

            muxer = new MediaMuxer(audioFile.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
            int trackIndex = muxer.addTrack(audioFormat);
            muxer.start();
            muxerStarted = true;

            extractor.selectTrack(audioTrackIndex);

            android.media.MediaCodec.BufferInfo bufferInfo = new android.media.MediaCodec.BufferInfo();
            ByteBuffer buffer = ByteBuffer.allocate(65536);

            while (true) {
                int sampleSize = extractor.readSampleData(buffer, 0);
                if (sampleSize < 0) break;

                bufferInfo.offset = 0;
                bufferInfo.size = sampleSize;
                bufferInfo.presentationTimeUs = extractor.getSampleTime();
                bufferInfo.flags = 0;

                muxer.writeSampleData(trackIndex, buffer, bufferInfo);
                extractor.advance();
            }

            return audioFile.getAbsolutePath();

        } finally {
            if (muxer != null) {
                try {
                    if (muxerStarted) muxer.stop();
                    muxer.release();
                } catch (Exception e) {
                    Log.w(TAG, "Error releasing MediaMuxer", e);
                }
            }
            if (extractor != null) {
                try {
                    extractor.release();
                } catch (Exception e) {
                    Log.w(TAG, "Error releasing MediaExtractor", e);
                }
            }
        }
    }

    private static String convertToWavWithFFmpeg(String inputPath, List<File> tempFiles) throws IOException {
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

        String command = String.format("-y -i \"%s\" -acodec pcm_s16le -ac 1 -ar 16000 \"%s\"",
                inputPath, wavFile.getAbsolutePath());
        FFmpegKit.execute(command);

        if (!wavFile.exists() || wavFile.length() == 0) {
            throw new IOException("FFmpeg conversion failed or produced empty file");
        }

        return wavFile.getAbsolutePath();
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
