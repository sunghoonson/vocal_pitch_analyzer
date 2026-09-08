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
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.AcousticEchoCanceler;
import android.media.audiofx.AutomaticGainControl;
import android.media.audiofx.NoiseSuppressor;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
            applyAudioEffects();
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
        // Deliberately do not stop. The foreground service is the actual mic
        // bridge and must survive the Activity/task being backgrounded.
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
                    new AudioRecord.Builder()
                            .setAudioSource(
                                    MediaRecorder.AudioSource.VOICE_RECOGNITION
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
                            500
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
                                "PC 연결됨 / 48kHz / "
                                        + effectsText(),
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

            String hello = String.format(
                    Locale.US,
                    "{"
                            + "\"type\":\"hello\","
                            + "\"sampleRate\":%d,"
                            + "\"packetFrames\":%d,"
                            + "\"deviceLabel\":\"S24 Native AudioRecord\","
                            + "\"trackSettings\":{"
                            + "\"source\":\"VOICE_RECOGNITION\","
                            + "\"screenOffForegroundService\":true"
                            + "},"
                            + "\"requestedNoiseSuppression\":%s,"
                            + "\"requestedEchoCancellation\":%s,"
                            + "\"requestedAutoGainControl\":%s"
                            + "}",
                    SAMPLE_RATE,
                    PACKET_FRAMES,
                    useNs
                            ? "true"
                            : "false",
                    useAec
                            ? "true"
                            : "false",
                    useAgc
                            ? "true"
                            : "false"
            );

            candidate.sendText(
                    hello
            );

            webSocket = candidate;
            connected = true;

            updateStatus(
                    "PC 연결됨 / 48kHz / "
                            + effectsText(),
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

    private String effectsText() {
        return "NS="
                + (
                useNs
                        ? "ON"
                        : "OFF"
        )
                + " / AEC="
                + (
                useAec
                        ? "ON"
                        : "OFF"
        )
                + " / AGC="
                + (
                useAgc
                        ? "ON"
                        : "OFF"
        );
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
