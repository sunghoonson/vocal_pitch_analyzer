package local.vocalpitch.s24mic;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.MicrophoneDirection;
import android.media.MicrophoneInfo;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;

import org.json.JSONObject;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Locale;

public final class MicForegroundService
        extends Service {

    static final String ACTION_START =
            "local.vocalpitch.s24mic.START";
    static final String ACTION_STOP =
            "local.vocalpitch.s24mic.STOP";

    static final String EXTRA_NS =
            "noise_suppression";
    static final String EXTRA_AEC =
            "echo_cancellation";
    static final String EXTRA_AGC =
            "auto_gain_control";
    static final String EXTRA_AUDIO_SOURCE =
            "audio_source";
    static final String EXTRA_DIRECTION =
            "microphone_direction";
    static final String EXTRA_FIELD_ZOOM =
            "microphone_field_zoom";

    static final int SAMPLE_RATE = 48000;
    static final int PACKET_FRAMES = 960;
    static final int NOTIFICATION_ID = 4401;

    private static final String CHANNEL_ID =
            "s24_native_mic";

    private volatile boolean running = false;
    private volatile boolean connected = false;

    private Thread captureThread;
    private AudioRecord audioRecord;
    private NoiseSuppressor noiseSuppressor;
    private AcousticEchoCanceler echoCanceler;
    private AutomaticGainControl autoGainControl;
    private WebSocketLite webSocket;
    private PowerManager.WakeLock cpuWakeLock;

    private boolean useNs = true;
    private boolean useAec = false;
    private boolean useAgc = false;

    private int requestedAudioSource =
            MediaRecorder.AudioSource.VOICE_RECOGNITION;
    private int actualAudioSource =
            MediaRecorder.AudioSource.VOICE_RECOGNITION;
    private int preferredDirection =
            MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER;
    private float microphoneFieldZoom = 0.75f;

    private boolean directionApplied = false;
    private boolean fieldApplied = false;
    private String activeMicrophoneSummary = "";

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(
            Intent intent,
            int flags,
            int startId
    ) {
        String action =
                intent != null
                        ? intent.getAction()
                        : null;

        if (ACTION_STOP.equals(action)) {
            stopBridge();
            stopSelf();
            return START_NOT_STICKY;
        }

        if (
                checkSelfPermission(
                        Manifest.permission.RECORD_AUDIO
                )
                        != PackageManager.PERMISSION_GRANTED
        ) {
            updateStatus(
                    "마이크 권한 없음",
                    false
            );
            stopSelf();
            return START_NOT_STICKY;
        }

        if (intent != null) {
            useNs = intent.getBooleanExtra(
                    EXTRA_NS,
                    true
            );
            useAec = intent.getBooleanExtra(
                    EXTRA_AEC,
                    false
            );
            useAgc = intent.getBooleanExtra(
                    EXTRA_AGC,
                    false
            );
            requestedAudioSource = sanitizeAudioSource(
                    intent.getIntExtra(
                            EXTRA_AUDIO_SOURCE,
                            MediaRecorder.AudioSource.VOICE_RECOGNITION
                    )
            );
            preferredDirection = sanitizeDirection(
                    intent.getIntExtra(
                            EXTRA_DIRECTION,
                            MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER
                    )
            );
            microphoneFieldZoom = clampZoom(
                    intent.getFloatExtra(
                            EXTRA_FIELD_ZOOM,
                            0.75f
                    )
            );
        }

        startForeground(
                NOTIFICATION_ID,
                buildNotification(
                        "PC 연결 준비 중..."
                )
        );

        acquireCpuWakeLock();

        if (!running) {
            startBridge();
        } else {
            // Source/direction changes require rebuilding AudioRecord.
            updateStatus(
                    "설정 변경은 Stop 후 Start에서 적용됩니다.",
                    connected
            );
        }

        return START_STICKY;
    }

    @Override
    public IBinder onBind(
            Intent intent
    ) {
        return null;
    }

    @Override
    public void onTaskRemoved(
            Intent rootIntent
    ) {
        super.onTaskRemoved(
                rootIntent
        );
    }

    @Override
    public void onDestroy() {
        stopBridge();
        super.onDestroy();
    }

    private void startBridge() {
        running = true;

        captureThread = new Thread(
                this::captureLoop,
                "S24Mic-Capture"
        );
        captureThread.start();

        updateStatus(
                "Native mic service 시작",
                false
        );
    }

    private void stopBridge() {
        running = false;
        connected = false;

        WebSocketLite ws = webSocket;
        webSocket = null;

        if (ws != null) {
            ws.close();
        }

        AudioRecord recorder = audioRecord;
        audioRecord = null;

        if (recorder != null) {
            try {
                recorder.stop();
            } catch (Exception ignored) {
            }

            try {
                recorder.release();
            } catch (Exception ignored) {
            }
        }

        releaseAudioEffects();
        releaseCpuWakeLock();

        Thread thread = captureThread;
        captureThread = null;

        if (
                thread != null
                        && thread != Thread.currentThread()
        ) {
            try {
                thread.join(
                        1000
                );
            } catch (InterruptedException ignored) {
                Thread.currentThread()
                        .interrupt();
            }
        }

        updateStatus(
                "중지됨",
                false
        );
    }

    private void captureLoop() {
        try {
            int minBytes =
                    AudioRecord.getMinBufferSize(
                            SAMPLE_RATE,
                            AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT
                    );

            if (minBytes <= 0) {
                throw new IllegalStateException(
                        "AudioRecord min buffer="
                                + minBytes
                );
            }

            int bufferBytes = Math.max(
                    minBytes
                            * 2,
                    PACKET_FRAMES
                            * 2
                            * 8
            );

            AudioRecord recorder =
                    buildRecorder(
                            requestedAudioSource,
                            bufferBytes
                    );

            if (
                    recorder.getState()
                            != AudioRecord.STATE_INITIALIZED
            ) {
                recorder.release();
                throw new IllegalStateException(
                        "AudioRecord initialization failed"
                );
            }

            audioRecord = recorder;

            directionApplied =
                    recorder.setPreferredMicrophoneDirection(
                            preferredDirection
                    );
            fieldApplied =
                    recorder.setPreferredMicrophoneFieldDimension(
                            microphoneFieldZoom
                    );

            applyAudioEffects();

            recorder.startRecording();

            if (
                    recorder.getRecordingState()
                            != AudioRecord.RECORDSTATE_RECORDING
            ) {
                throw new IllegalStateException(
                        "AudioRecord start failed"
                );
            }

            SystemClock.sleep(
                    80
            );

            activeMicrophoneSummary =
                    describeActiveMicrophones(
                            recorder
                    );

            persistNativeMicState();

            short[] input = new short[
                    PACKET_FRAMES
                    ];

            ByteBuffer floatPacket =
                    ByteBuffer.allocate(
                                    PACKET_FRAMES
                                            * 4
                            )
                            .order(
                                    ByteOrder.LITTLE_ENDIAN
                            );

            while (running) {
                ensureConnected();

                if (
                        webSocket == null
                                || !webSocket.isOpen()
                ) {
                    SystemClock.sleep(
                            250
                    );
                    continue;
                }

                int count = recorder.read(
                        input,
                        0,
                        input.length,
                        AudioRecord.READ_BLOCKING
                );

                if (count < 0) {
                    throw new IllegalStateException(
                            "AudioRecord read="
                                    + count
                    );
                }

                if (count == 0) {
                    continue;
                }

                floatPacket.clear();

                for (
                        int i = 0;
                        i < count;
                        i++
                ) {
                    floatPacket.putFloat(
                            input[i]
                                    / 32768.0f
                    );
                }

                byte[] packet = new byte[
                        count
                                * 4
                        ];
                floatPacket.flip();
                floatPacket.get(
                        packet
                );

                try {
                    webSocket.sendBinary(
                            packet
                    );

                    if (!connected) {
                        connected = true;
                        updateStatus(
                                connectedStatusText(),
                                true
                        );
                    }

                } catch (Exception exc) {
                    connected = false;

                    WebSocketLite failed =
                            webSocket;
                    webSocket = null;

                    if (failed != null) {
                        failed.close();
                    }

                    updateStatus(
                            "PC 재연결 중: "
                                    + exc.getClass()
                                    .getSimpleName(),
                            false
                    );
                }
            }

        } catch (Exception exc) {
            updateStatus(
                    "ERROR: "
                            + exc.getClass()
                            .getSimpleName()
                            + ": "
                            + exc.getMessage(),
                    false
            );

        } finally {
            running = false;
            connected = false;

            AudioRecord recorder =
                    audioRecord;
            audioRecord = null;

            if (recorder != null) {
                try {
                    recorder.stop();
                } catch (Exception ignored) {
                }

                try {
                    recorder.release();
                } catch (Exception ignored) {
                }
            }

            releaseAudioEffects();
        }
    }

    private AudioRecord buildRecorder(
            int requestedSource,
            int bufferBytes
    ) {
        int candidate = requestedSource;

        if (
                candidate
                        == MediaRecorder.AudioSource.UNPROCESSED
                && !supportsUnprocessed()
        ) {
            candidate =
                    MediaRecorder.AudioSource.VOICE_RECOGNITION;
        }

        try {
            AudioRecord recorder =
                    buildRecorderOnce(
                            candidate,
                            bufferBytes
                    );
            actualAudioSource = candidate;
            return recorder;

        } catch (Exception first) {
            if (
                    candidate
                            == MediaRecorder.AudioSource.VOICE_RECOGNITION
            ) {
                throw first;
            }

            AudioRecord fallback =
                    buildRecorderOnce(
                            MediaRecorder.AudioSource.VOICE_RECOGNITION,
                            bufferBytes
                    );
            actualAudioSource =
                    MediaRecorder.AudioSource.VOICE_RECOGNITION;
            return fallback;
        }
    }

    private AudioRecord buildRecorderOnce(
            int source,
            int bufferBytes
    ) {
        return new AudioRecord.Builder()
                .setAudioSource(
                        source
                )
                .setAudioFormat(
                        new AudioFormat.Builder()
                                .setEncoding(
                                        AudioFormat.ENCODING_PCM_16BIT
                                )
                                .setSampleRate(
                                        SAMPLE_RATE
                                )
                                .setChannelMask(
                                        AudioFormat.CHANNEL_IN_MONO
                                )
                                .build()
                )
                .setBufferSizeInBytes(
                        bufferBytes
                )
                .build();
    }

    private boolean supportsUnprocessed() {
        try {
            AudioManager manager =
                    (AudioManager) getSystemService(
                            Context.AUDIO_SERVICE
                    );

            String value =
                    manager.getProperty(
                            AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED
                    );

            return Boolean.parseBoolean(
                    value
            );

        } catch (Exception ignored) {
            return false;
        }
    }

    private void ensureConnected() {
        if (
                webSocket != null
                        && webSocket.isOpen()
        ) {
            return;
        }

        WebSocketLite candidate =
                new WebSocketLite();

        try {
            updateStatus(
                    "PC WebSocket 연결 중...",
                    false
            );

            candidate.connect(
                    "127.0.0.1",
                    8791,
                    "/"
            );

            candidate.sendText(
                    buildHelloJson()
            );

            webSocket = candidate;
            connected = true;

            updateStatus(
                    connectedStatusText(),
                    true
            );

        } catch (Exception exc) {
            candidate.close();
            connected = false;

            updateStatus(
                    "PC 연결 대기 / "
                            + exc.getClass()
                            .getSimpleName(),
                    false
            );

            SystemClock.sleep(
                    1000
            );
        }
    }

    private String buildHelloJson() {
        try {
            JSONObject root =
                    new JSONObject();

            root.put(
                    "type",
                    "hello"
            );
            root.put(
                    "sampleRate",
                    SAMPLE_RATE
            );
            root.put(
                    "packetFrames",
                    PACKET_FRAMES
            );
            root.put(
                    "deviceLabel",
                    "S24 Native AudioRecord v4.6"
            );

            JSONObject track =
                    new JSONObject();

            track.put(
                    "source",
                    audioSourceName(
                            actualAudioSource
                    )
            );
            track.put(
                    "screenOffForegroundService",
                    true
            );
            root.put(
                    "trackSettings",
                    track
            );

            root.put(
                    "requestedNoiseSuppression",
                    useNs
            );
            root.put(
                    "requestedEchoCancellation",
                    useAec
            );
            root.put(
                    "requestedAutoGainControl",
                    useAgc
            );

            root.put(
                    "actualNoiseSuppression",
                    actualNoiseSuppression()
            );
            root.put(
                    "actualEchoCancellation",
                    actualEchoCancellation()
            );
            root.put(
                    "actualAutoGainControl",
                    actualAutoGainControl()
            );

            root.put(
                    "audioSourceName",
                    audioSourceName(
                            actualAudioSource
                    )
            );
            root.put(
                    "microphoneDirectionName",
                    directionName(
                            preferredDirection
                    )
            );
            root.put(
                    "microphoneFieldZoom",
                    microphoneFieldZoom
            );
            root.put(
                    "directionApplied",
                    directionApplied
            );
            root.put(
                    "fieldApplied",
                    fieldApplied
            );
            root.put(
                    "activeMicrophones",
                    activeMicrophoneSummary
            );

            return root.toString();

        } catch (Exception exc) {
            return "{"
                    + "\"type\":\"hello\","
                    + "\"sampleRate\":48000,"
                    + "\"packetFrames\":960,"
                    + "\"deviceLabel\":\"S24 Native AudioRecord v4.6\""
                    + "}";
        }
    }

    private synchronized void applyAudioEffects() {
        AudioRecord recorder =
                audioRecord;

        if (recorder == null) {
            return;
        }

        releaseAudioEffects();

        int session = recorder.getAudioSessionId();

        try {
            if (
                    NoiseSuppressor.isAvailable()
            ) {
                noiseSuppressor =
                        NoiseSuppressor.create(
                                session
                        );

                if (noiseSuppressor != null) {
                    noiseSuppressor.setEnabled(
                            useNs
                    );
                }
            }
        } catch (Exception ignored) {
        }

        try {
            if (
                    AcousticEchoCanceler.isAvailable()
            ) {
                echoCanceler =
                        AcousticEchoCanceler.create(
                                session
                        );

                if (echoCanceler != null) {
                    echoCanceler.setEnabled(
                            useAec
                    );
                }
            }
        } catch (Exception ignored) {
        }

        try {
            if (
                    AutomaticGainControl.isAvailable()
            ) {
                autoGainControl =
                        AutomaticGainControl.create(
                                session
                        );

                if (autoGainControl != null) {
                    autoGainControl.setEnabled(
                            useAgc
                    );
                }
            }
        } catch (Exception ignored) {
        }
    }

    private synchronized void releaseAudioEffects() {
        if (noiseSuppressor != null) {
            try {
                noiseSuppressor.release();
            } catch (Exception ignored) {
            }
            noiseSuppressor = null;
        }

        if (echoCanceler != null) {
            try {
                echoCanceler.release();
            } catch (Exception ignored) {
            }
            echoCanceler = null;
        }

        if (autoGainControl != null) {
            try {
                autoGainControl.release();
            } catch (Exception ignored) {
            }
            autoGainControl = null;
        }
    }

    private boolean actualNoiseSuppression() {
        try {
            return noiseSuppressor != null
                    && noiseSuppressor.getEnabled();
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean actualEchoCancellation() {
        try {
            return echoCanceler != null
                    && echoCanceler.getEnabled();
        } catch (Exception ignored) {
            return false;
        }
    }

    private boolean actualAutoGainControl() {
        try {
            return autoGainControl != null
                    && autoGainControl.getEnabled();
        } catch (Exception ignored) {
            return false;
        }
    }

    private String effectsText() {
        return "NS="
                + (
                actualNoiseSuppression()
                        ? "ON"
                        : "OFF"
        )
                + " / AEC="
                + (
                actualEchoCancellation()
                        ? "ON"
                        : "OFF"
        )
                + " / AGC="
                + (
                actualAutoGainControl()
                        ? "ON"
                        : "OFF"
        );
    }

    private String connectedStatusText() {
        return "PC 연결됨 / 48kHz / "
                + audioSourceName(
                actualAudioSource
        )
                + " / "
                + directionName(
                preferredDirection
        )
                + String.format(
                Locale.US,
                " / focus=%+.2f / ",
                microphoneFieldZoom
        )
                + effectsText();
    }

    private String describeActiveMicrophones(
            AudioRecord recorder
    ) {
        try {
            List<MicrophoneInfo> microphones =
                    recorder.getActiveMicrophones();

            if (
                    microphones == null
                            || microphones.isEmpty()
            ) {
                return "none/reported-empty";
            }

            StringBuilder result =
                    new StringBuilder();

            for (
                    int i = 0;
                    i < microphones.size();
                    i++
            ) {
                MicrophoneInfo mic =
                        microphones.get(
                                i
                        );

                if (i > 0) {
                    result.append(
                            " | "
                    );
                }

                result.append(
                        "#"
                );
                result.append(
                        i
                );
                result.append(
                        " id="
                );
                result.append(
                        mic.getId()
                );
                result.append(
                        " "
                );
                result.append(
                        directionalityName(
                                mic.getDirectionality()
                        )
                );
                result.append(
                        " group="
                );
                result.append(
                        mic.getGroup()
                );
                result.append(
                        "/"
                );
                result.append(
                        mic.getIndexInTheGroup()
                );
                result.append(
                        " map="
                );
                result.append(
                        mic.getChannelMapping()
                );
            }

            return result.toString();

        } catch (Exception exc) {
            return "query-error:"
                    + exc.getClass()
                    .getSimpleName();
        }
    }

    private void persistNativeMicState() {
        getSharedPreferences(
                "state",
                MODE_PRIVATE
        )
                .edit()
                .putString(
                        "audio_source",
                        audioSourceName(
                                actualAudioSource
                        )
                )
                .putString(
                        "direction",
                        directionName(
                                preferredDirection
                        )
                )
                .putFloat(
                        "field_zoom",
                        microphoneFieldZoom
                )
                .putBoolean(
                        "direction_applied",
                        directionApplied
                )
                .putBoolean(
                        "field_applied",
                        fieldApplied
                )
                .putString(
                        "active_mics",
                        activeMicrophoneSummary
                )
                .putBoolean(
                        "actual_ns",
                        actualNoiseSuppression()
                )
                .putBoolean(
                        "actual_aec",
                        actualEchoCancellation()
                )
                .putBoolean(
                        "actual_agc",
                        actualAutoGainControl()
                )
                .apply();
    }

    private static int sanitizeAudioSource(
            int value
    ) {
        if (
                value
                        == MediaRecorder.AudioSource.MIC
                || value
                        == MediaRecorder.AudioSource.VOICE_RECOGNITION
                || value
                        == MediaRecorder.AudioSource.VOICE_COMMUNICATION
                || value
                        == MediaRecorder.AudioSource.UNPROCESSED
                || value
                        == MediaRecorder.AudioSource.VOICE_PERFORMANCE
        ) {
            return value;
        }

        return MediaRecorder.AudioSource.VOICE_RECOGNITION;
    }

    private static int sanitizeDirection(
            int value
    ) {
        if (
                value
                        == MicrophoneDirection.MIC_DIRECTION_UNSPECIFIED
                || value
                        == MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER
                || value
                        == MicrophoneDirection.MIC_DIRECTION_AWAY_FROM_USER
        ) {
            return value;
        }

        return MicrophoneDirection.MIC_DIRECTION_UNSPECIFIED;
    }

    private static float clampZoom(
            float value
    ) {
        return Math.max(
                -1.0f,
                Math.min(
                        value,
                        1.0f
                )
        );
    }

    static String audioSourceName(
            int source
    ) {
        if (
                source
                        == MediaRecorder.AudioSource.VOICE_PERFORMANCE
        ) {
            return "VOICE_PERFORMANCE";
        }
        if (
                source
                        == MediaRecorder.AudioSource.UNPROCESSED
        ) {
            return "UNPROCESSED";
        }
        if (
                source
                        == MediaRecorder.AudioSource.VOICE_COMMUNICATION
        ) {
            return "VOICE_COMMUNICATION";
        }
        if (
                source
                        == MediaRecorder.AudioSource.MIC
        ) {
            return "MIC";
        }

        return "VOICE_RECOGNITION";
    }

    static String directionName(
            int direction
    ) {
        if (
                direction
                        == MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER
        ) {
            return "TOWARDS_USER";
        }
        if (
                direction
                        == MicrophoneDirection.MIC_DIRECTION_AWAY_FROM_USER
        ) {
            return "AWAY_FROM_USER";
        }

        return "UNSPECIFIED";
    }

    static String directionalityName(
            int directionality
    ) {
        if (
                directionality
                        == MicrophoneInfo.DIRECTIONALITY_OMNI
        ) {
            return "OMNI";
        }
        if (
                directionality
                        == MicrophoneInfo.DIRECTIONALITY_BI_DIRECTIONAL
        ) {
            return "BI";
        }
        if (
                directionality
                        == MicrophoneInfo.DIRECTIONALITY_CARDIOID
        ) {
            return "CARDIOID";
        }
        if (
                directionality
                        == MicrophoneInfo.DIRECTIONALITY_HYPER_CARDIOID
        ) {
            return "HYPER_CARDIOID";
        }
        if (
                directionality
                        == MicrophoneInfo.DIRECTIONALITY_SUPER_CARDIOID
        ) {
            return "SUPER_CARDIOID";
        }

        return "UNKNOWN";
    }

    private void acquireCpuWakeLock() {
        if (
                cpuWakeLock != null
                        && cpuWakeLock.isHeld()
        ) {
            return;
        }

        PowerManager powerManager =
                (PowerManager) getSystemService(
                        Context.POWER_SERVICE
                );

        cpuWakeLock =
                powerManager.newWakeLock(
                        PowerManager.PARTIAL_WAKE_LOCK,
                        "VPA:S24NativeMic"
                );

        cpuWakeLock.setReferenceCounted(
                false
        );
        cpuWakeLock.acquire();
    }

    private void releaseCpuWakeLock() {
        if (
                cpuWakeLock != null
                        && cpuWakeLock.isHeld()
        ) {
            try {
                cpuWakeLock.release();
            } catch (Exception ignored) {
            }
        }

        cpuWakeLock = null;
    }

    private void createNotificationChannel() {
        NotificationManager manager =
                getSystemService(
                        NotificationManager.class
                );

        NotificationChannel channel =
                new NotificationChannel(
                        CHANNEL_ID,
                        "S24 Native Mic",
                        NotificationManager.IMPORTANCE_LOW
                );
        channel.setDescription(
                "화면이 꺼져도 계속 동작하는 S24 microphone bridge"
        );

        manager.createNotificationChannel(
                channel
        );
    }

    private Notification buildNotification(
            String text
    ) {
        Intent activityIntent =
                new Intent(
                        this,
                        MainActivity.class
                );

        PendingIntent activityPending =
                PendingIntent.getActivity(
                        this,
                        10,
                        activityIntent,
                        PendingIntent.FLAG_UPDATE_CURRENT
                                | PendingIntent.FLAG_IMMUTABLE
                );

        Intent stopIntent =
                new Intent(
                        this,
                        MicForegroundService.class
                );
        stopIntent.setAction(
                ACTION_STOP
        );

        PendingIntent stopPending =
                PendingIntent.getService(
                        this,
                        11,
                        stopIntent,
                        PendingIntent.FLAG_UPDATE_CURRENT
                                | PendingIntent.FLAG_IMMUTABLE
                );

        return new Notification.Builder(
                this,
                CHANNEL_ID
        )
                .setContentTitle(
                        "VPA S24 Native Mic"
                )
                .setContentText(
                        text
                )
                .setSmallIcon(
                        android.R.drawable.ic_btn_speak_now
                )
                .setContentIntent(
                        activityPending
                )
                .setOngoing(
                        true
                )
                .addAction(
                        new Notification.Action.Builder(
                                null,
                                "Stop",
                                stopPending
                        ).build()
                )
                .build();
    }

    private void updateStatus(
            String text,
            boolean isConnected
    ) {
        connected = isConnected;

        getSharedPreferences(
                "state",
                MODE_PRIVATE
        )
                .edit()
                .putBoolean(
                        "running",
                        running
                )
                .putBoolean(
                        "connected",
                        connected
                )
                .putString(
                        "status",
                        text
                )
                .putLong(
                        "updated",
                        System.currentTimeMillis()
                )
                .apply();

        if (running) {
            NotificationManager manager =
                    getSystemService(
                            NotificationManager.class
                    );

            manager.notify(
                    NOTIFICATION_ID,
                    buildNotification(
                            text
                    )
            );
        }
    }
}
