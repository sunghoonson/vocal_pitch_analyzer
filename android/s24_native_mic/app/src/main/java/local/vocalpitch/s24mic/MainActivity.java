package local.vocalpitch.s24mic;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.MediaRecorder;
import android.media.MicrophoneDirection;
import android.media.MicrophoneInfo;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.List;
import java.util.Locale;

public final class MainActivity
        extends Activity {

    private static final int REQUEST_AUDIO = 1001;
    private static final int REQUEST_NOTIFICATIONS = 1002;

    private final Handler handler =
            new Handler(
                    Looper.getMainLooper()
            );

    private TextView statusView;
    private TextView focusValueView;
    private TextView diagnosticsView;

    private CheckBox nsCheck;
    private CheckBox aecCheck;
    private CheckBox agcCheck;

    private Spinner sourceSpinner;
    private Spinner directionSpinner;
    private SeekBar fieldSeek;

    private Button startButton;
    private Button stopButton;

    private boolean pendingAutoStart = false;

    private static final String[] SOURCE_LABELS = new String[] {
            "Voice Focus / Recognition",
            "Low Latency / Voice Performance",
            "Raw Diagnostic / Unprocessed",
            "VoIP / Voice Communication",
            "Plain MIC"
    };

    private static final int[] SOURCE_VALUES = new int[] {
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            MediaRecorder.AudioSource.VOICE_PERFORMANCE,
            MediaRecorder.AudioSource.UNPROCESSED,
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            MediaRecorder.AudioSource.MIC
    };

    private static final String[] DIRECTION_LABELS = new String[] {
            "Towards User (화면 방향)",
            "Unspecified / Default",
            "Away From User (후면 방향)"
    };

    private static final int[] DIRECTION_VALUES = new int[] {
            MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER,
            MicrophoneDirection.MIC_DIRECTION_UNSPECIFIED,
            MicrophoneDirection.MIC_DIRECTION_AWAY_FROM_USER
    };

    private final Runnable statusPoll =
            new Runnable() {
                @Override
                public void run() {
                    refreshStatus();
                    handler.postDelayed(
                            this,
                            500
                    );
                }
            };

    @Override
    protected void onCreate(
            Bundle savedInstanceState
    ) {
        super.onCreate(
                savedInstanceState
        );

        pendingAutoStart =
                getIntent() != null
                        && getIntent()
                        .getBooleanExtra(
                                "auto_start",
                                false
                        );

        buildUi();
        loadPreferences();
        refreshDiagnostics();

        handler.post(
                statusPoll
        );

        if (pendingAutoStart) {
            ensurePermissionThenStart();
        }
    }

    @Override
    protected void onNewIntent(
            Intent intent
    ) {
        super.onNewIntent(
                intent
        );
        setIntent(
                intent
        );

        if (
                intent != null
                        && intent.getBooleanExtra(
                        "auto_start",
                        false
                )
        ) {
            pendingAutoStart = true;
            ensurePermissionThenStart();
        }
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacks(
                statusPoll
        );
        super.onDestroy();
    }

    private void buildUi() {
        ScrollView scrollView =
                new ScrollView(
                        this
                );

        LinearLayout root =
                new LinearLayout(
                        this
                );
        root.setOrientation(
                LinearLayout.VERTICAL
        );
        root.setPadding(
                42,
                42,
                42,
                42
        );

        TextView title =
                new TextView(
                        this
                );
        title.setText(
                "VPA S24 Native Mic v4.6"
        );
        title.setTextSize(
                26
        );
        title.setTextColor(
                Color.BLACK
        );
        root.addView(
                title,
                fullWidth()
        );

        TextView description =
                new TextView(
                        this
                );
        description.setText(
                "Android AudioRecord + Foreground Service 기반입니다.\n"
                        + "v4.6은 S24의 오디오 HAL에 microphone direction/field focus를 요청하고, "
                        + "Voice Recognition / Voice Performance / Unprocessed 경로를 직접 비교할 수 있습니다.\n"
                        + "Direction/Focus는 하드웨어 카디오이드와 동일하지 않으며, "
                        + "Samsung HAL이 지원하는 범위에서만 실제 효과가 납니다."
        );
        description.setTextSize(
                15
        );
        description.setPadding(
                0,
                18,
                0,
                18
        );
        root.addView(
                description,
                fullWidth()
        );

        statusView =
                new TextView(
                        this
                );
        statusView.setTextSize(
                16
        );
        statusView.setPadding(
                20,
                20,
                20,
                20
        );
        statusView.setBackgroundColor(
                0xFFECEFF1
        );
        root.addView(
                statusView,
                fullWidth()
        );

        TextView sourceLabel =
                new TextView(
                        this
                );
        sourceLabel.setText(
                "Capture Profile"
        );
        sourceLabel.setTextSize(
                16
        );
        root.addView(
                sourceLabel,
                fullWidth()
        );

        sourceSpinner =
                new Spinner(
                        this
                );
        ArrayAdapter<String> sourceAdapter =
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_item,
                        SOURCE_LABELS
                );
        sourceAdapter.setDropDownViewResource(
                android.R.layout.simple_spinner_dropdown_item
        );
        sourceSpinner.setAdapter(
                sourceAdapter
        );
        root.addView(
                sourceSpinner,
                fullWidth()
        );

        TextView directionLabel =
                new TextView(
                        this
                );
        directionLabel.setText(
                "Microphone Direction"
        );
        directionLabel.setTextSize(
                16
        );
        root.addView(
                directionLabel,
                fullWidth()
        );

        directionSpinner =
                new Spinner(
                        this
                );
        ArrayAdapter<String> directionAdapter =
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_item,
                        DIRECTION_LABELS
                );
        directionAdapter.setDropDownViewResource(
                android.R.layout.simple_spinner_dropdown_item
        );
        directionSpinner.setAdapter(
                directionAdapter
        );
        root.addView(
                directionSpinner,
                fullWidth()
        );

        focusValueView =
                new TextView(
                        this
                );
        focusValueView.setTextSize(
                16
        );
        root.addView(
                focusValueView,
                fullWidth()
        );

        fieldSeek =
                new SeekBar(
                        this
                );
        fieldSeek.setMax(
                200
        );
        fieldSeek.setProgress(
                175
        );
        fieldSeek.setOnSeekBarChangeListener(
                new SeekBar.OnSeekBarChangeListener() {
                    @Override
                    public void onProgressChanged(
                            SeekBar seekBar,
                            int progress,
                            boolean fromUser
                    ) {
                        updateFocusLabel();
                    }

                    @Override
                    public void onStartTrackingTouch(
                            SeekBar seekBar
                    ) {
                    }

                    @Override
                    public void onStopTrackingTouch(
                            SeekBar seekBar
                    ) {
                    }
                }
        );
        root.addView(
                fieldSeek,
                fullWidth()
        );

        LinearLayout presetRow =
                new LinearLayout(
                        this
                );
        presetRow.setOrientation(
                LinearLayout.HORIZONTAL
        );

        Button voiceFocusPreset =
                new Button(
                        this
                );
        voiceFocusPreset.setText(
                "Voice Focus"
        );
        voiceFocusPreset.setOnClickListener(
                view -> applyVoiceFocusPreset()
        );
        presetRow.addView(
                voiceFocusPreset,
                weighted()
        );

        Button lowLatencyPreset =
                new Button(
                        this
                );
        lowLatencyPreset.setText(
                "Low Latency"
        );
        lowLatencyPreset.setOnClickListener(
                view -> applyLowLatencyPreset()
        );
        presetRow.addView(
                lowLatencyPreset,
                weighted()
        );

        Button rawPreset =
                new Button(
                        this
                );
        rawPreset.setText(
                "Raw Test"
        );
        rawPreset.setOnClickListener(
                view -> applyRawPreset()
        );
        presetRow.addView(
                rawPreset,
                weighted()
        );

        root.addView(
                presetRow,
                fullWidth()
        );

        nsCheck =
                new CheckBox(
                        this
                );
        nsCheck.setText(
                "Android Noise Suppression"
        );
        root.addView(
                nsCheck,
                fullWidth()
        );

        aecCheck =
                new CheckBox(
                        this
                );
        aecCheck.setText(
                "Echo Cancellation"
        );
        root.addView(
                aecCheck,
                fullWidth()
        );

        agcCheck =
                new CheckBox(
                        this
                );
        agcCheck.setText(
                "Android Auto Gain Control"
        );
        root.addView(
                agcCheck,
                fullWidth()
        );

        TextView effectNote =
                new TextView(
                        this
                );
        effectNote.setText(
                "권장 시작점: Voice Focus + Towards User + Focus +0.75 + NS ON + AEC/AGC OFF.\n"
                        + "RVC 비교용 Raw Test에서는 Android 전처리를 최소화해 PC DSP/NVIDIA Broadcast와 역할이 겹치는지 확인할 수 있습니다."
        );
        effectNote.setTextSize(
                14
        );
        root.addView(
                effectNote,
                fullWidth()
        );

        startButton =
                new Button(
                        this
                );
        startButton.setText(
                "Start native mic"
        );
        startButton.setOnClickListener(
                view -> {
                    pendingAutoStart = true;
                    ensurePermissionThenStart();
                }
        );
        root.addView(
                startButton,
                fullWidth()
        );

        stopButton =
                new Button(
                        this
                );
        stopButton.setText(
                "Stop"
        );
        stopButton.setOnClickListener(
                view -> stopMicService()
        );
        root.addView(
                stopButton,
                fullWidth()
        );

        Button diagnosticButton =
                new Button(
                        this
                );
        diagnosticButton.setText(
                "S24 마이크 하드웨어/활성 마이크 진단 새로고침"
        );
        diagnosticButton.setOnClickListener(
                view -> refreshDiagnostics()
        );
        root.addView(
                diagnosticButton,
                fullWidth()
        );

        diagnosticsView =
                new TextView(
                        this
                );
        diagnosticsView.setTextSize(
                13
        );
        diagnosticsView.setTextIsSelectable(
                true
        );
        diagnosticsView.setPadding(
                16,
                16,
                16,
                16
        );
        diagnosticsView.setBackgroundColor(
                0xFFF5F5F5
        );
        root.addView(
                diagnosticsView,
                fullWidth()
        );

        Button batteryButton =
                new Button(
                        this
                );
        batteryButton.setText(
                "배터리 설정 열기"
        );
        batteryButton.setOnClickListener(
                view -> openBatterySettings()
        );
        root.addView(
                batteryButton,
                fullWidth()
        );

        scrollView.addView(
                root
        );

        setContentView(
                scrollView
        );
    }

    private LinearLayout.LayoutParams fullWidth() {
        LinearLayout.LayoutParams params =
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                );
        params.setMargins(
                0,
                8,
                0,
                8
        );
        return params;
    }

    private LinearLayout.LayoutParams weighted() {
        LinearLayout.LayoutParams params =
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1.0f
                );
        params.setMargins(
                4,
                4,
                4,
                4
        );
        return params;
    }

    private void loadPreferences() {
        SharedPreferences prefs =
                getSharedPreferences(
                        "options",
                        MODE_PRIVATE
                );

        nsCheck.setChecked(
                prefs.getBoolean(
                        "ns",
                        true
                )
        );
        aecCheck.setChecked(
                prefs.getBoolean(
                        "aec",
                        false
                )
        );
        agcCheck.setChecked(
                prefs.getBoolean(
                        "agc",
                        false
                )
        );

        setSpinnerForValue(
                sourceSpinner,
                SOURCE_VALUES,
                prefs.getInt(
                        "audio_source",
                        MediaRecorder.AudioSource.VOICE_RECOGNITION
                )
        );

        setSpinnerForValue(
                directionSpinner,
                DIRECTION_VALUES,
                prefs.getInt(
                        "direction",
                        MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER
                )
        );

        float zoom =
                prefs.getFloat(
                        "field_zoom",
                        0.75f
                );
        fieldSeek.setProgress(
                Math.round(
                        (zoom + 1.0f)
                                * 100.0f
                )
        );
        updateFocusLabel();
    }

    private void savePreferences() {
        getSharedPreferences(
                "options",
                MODE_PRIVATE
        )
                .edit()
                .putBoolean(
                        "ns",
                        nsCheck.isChecked()
                )
                .putBoolean(
                        "aec",
                        aecCheck.isChecked()
                )
                .putBoolean(
                        "agc",
                        agcCheck.isChecked()
                )
                .putInt(
                        "audio_source",
                        selectedSource()
                )
                .putInt(
                        "direction",
                        selectedDirection()
                )
                .putFloat(
                        "field_zoom",
                        selectedFieldZoom()
                )
                .apply();
    }

    private void applyVoiceFocusPreset() {
        setSpinnerForValue(
                sourceSpinner,
                SOURCE_VALUES,
                MediaRecorder.AudioSource.VOICE_RECOGNITION
        );
        setSpinnerForValue(
                directionSpinner,
                DIRECTION_VALUES,
                MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER
        );
        fieldSeek.setProgress(
                175
        );
        nsCheck.setChecked(
                true
        );
        aecCheck.setChecked(
                false
        );
        agcCheck.setChecked(
                false
        );
        updateFocusLabel();
        savePreferences();
    }

    private void applyLowLatencyPreset() {
        setSpinnerForValue(
                sourceSpinner,
                SOURCE_VALUES,
                MediaRecorder.AudioSource.VOICE_PERFORMANCE
        );
        setSpinnerForValue(
                directionSpinner,
                DIRECTION_VALUES,
                MicrophoneDirection.MIC_DIRECTION_TOWARDS_USER
        );
        fieldSeek.setProgress(
                150
        );
        nsCheck.setChecked(
                true
        );
        aecCheck.setChecked(
                false
        );
        agcCheck.setChecked(
                false
        );
        updateFocusLabel();
        savePreferences();
    }

    private void applyRawPreset() {
        setSpinnerForValue(
                sourceSpinner,
                SOURCE_VALUES,
                MediaRecorder.AudioSource.UNPROCESSED
        );
        setSpinnerForValue(
                directionSpinner,
                DIRECTION_VALUES,
                MicrophoneDirection.MIC_DIRECTION_UNSPECIFIED
        );
        fieldSeek.setProgress(
                100
        );
        nsCheck.setChecked(
                false
        );
        aecCheck.setChecked(
                false
        );
        agcCheck.setChecked(
                false
        );
        updateFocusLabel();
        savePreferences();
    }

    private int selectedSource() {
        int index =
                sourceSpinner.getSelectedItemPosition();

        if (
                index < 0
                        || index >= SOURCE_VALUES.length
        ) {
            return MediaRecorder.AudioSource.VOICE_RECOGNITION;
        }

        return SOURCE_VALUES[
                index
                ];
    }

    private int selectedDirection() {
        int index =
                directionSpinner.getSelectedItemPosition();

        if (
                index < 0
                        || index >= DIRECTION_VALUES.length
        ) {
            return MicrophoneDirection.MIC_DIRECTION_UNSPECIFIED;
        }

        return DIRECTION_VALUES[
                index
                ];
    }

    private float selectedFieldZoom() {
        return (
                fieldSeek.getProgress()
                        / 100.0f
        ) - 1.0f;
    }

    private void updateFocusLabel() {
        if (focusValueView == null) {
            return;
        }

        float value =
                selectedFieldZoom();

        focusValueView.setText(
                String.format(
                        Locale.US,
                        "Mic Field Focus: %+.2f   (-1.0 Wide / 0 Normal / +1.0 Max Focus)",
                        value
                )
        );
    }

    private static void setSpinnerForValue(
            Spinner spinner,
            int[] values,
            int wanted
    ) {
        for (
                int i = 0;
                i < values.length;
                i++
        ) {
            if (
                    values[i]
                            == wanted
            ) {
                spinner.setSelection(
                        i
                );
                return;
            }
        }

        spinner.setSelection(
                0
        );
    }

    private void ensurePermissionThenStart() {
        if (
                checkSelfPermission(
                        Manifest.permission.RECORD_AUDIO
                )
                        != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                    new String[] {
                            Manifest.permission.RECORD_AUDIO
                    },
                    REQUEST_AUDIO
            );
            return;
        }

        requestNotificationPermissionIfNeeded();
        startMicService();
    }

    private void requestNotificationPermissionIfNeeded() {
        if (
                android.os.Build.VERSION.SDK_INT
                        >= 33
                && checkSelfPermission(
                        Manifest.permission.POST_NOTIFICATIONS
                )
                        != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                    new String[] {
                            Manifest.permission.POST_NOTIFICATIONS
                    },
                    REQUEST_NOTIFICATIONS
            );
        }
    }

    private void startMicService() {
        savePreferences();

        Intent intent =
                new Intent(
                        this,
                        MicForegroundService.class
                );
        intent.setAction(
                MicForegroundService.ACTION_START
        );
        intent.putExtra(
                MicForegroundService.EXTRA_NS,
                nsCheck.isChecked()
        );
        intent.putExtra(
                MicForegroundService.EXTRA_AEC,
                aecCheck.isChecked()
        );
        intent.putExtra(
                MicForegroundService.EXTRA_AGC,
                agcCheck.isChecked()
        );
        intent.putExtra(
                MicForegroundService.EXTRA_AUDIO_SOURCE,
                selectedSource()
        );
        intent.putExtra(
                MicForegroundService.EXTRA_DIRECTION,
                selectedDirection()
        );
        intent.putExtra(
                MicForegroundService.EXTRA_FIELD_ZOOM,
                selectedFieldZoom()
        );

        startForegroundService(
                intent
        );

        pendingAutoStart = false;
    }

    private void stopMicService() {
        Intent intent =
                new Intent(
                        this,
                        MicForegroundService.class
                );
        intent.setAction(
                MicForegroundService.ACTION_STOP
        );
        startService(
                intent
        );
    }

    private void refreshStatus() {
        SharedPreferences state =
                getSharedPreferences(
                        "state",
                        MODE_PRIVATE
                );

        boolean running =
                state.getBoolean(
                        "running",
                        false
                );
        boolean connected =
                state.getBoolean(
                        "connected",
                        false
                );
        String text =
                state.getString(
                        "status",
                        "대기 중"
                );
        String source =
                state.getString(
                        "audio_source",
                        "-"
                );
        String direction =
                state.getString(
                        "direction",
                        "-"
                );
        float focus =
                state.getFloat(
                        "field_zoom",
                        0.0f
                );
        boolean directionApplied =
                state.getBoolean(
                        "direction_applied",
                        false
                );
        boolean fieldApplied =
                state.getBoolean(
                        "field_applied",
                        false
                );

        statusView.setText(
                "Service: "
                        + (
                        running
                                ? "RUNNING"
                                : "STOPPED"
                )
                        + "\nPC: "
                        + (
                        connected
                                ? "CONNECTED"
                                : "WAITING"
                )
                        + "\nSource: "
                        + source
                        + "\nDirection: "
                        + direction
                        + " / API="
                        + (
                        directionApplied
                                ? "accepted"
                                : "not-confirmed"
                )
                        + String.format(
                        Locale.US,
                        "\nField Focus: %+.2f / API=%s",
                        focus,
                        fieldApplied
                                ? "accepted"
                                : "not-confirmed"
                )
                        + "\n"
                        + text
        );

        startButton.setEnabled(
                !running
        );
        stopButton.setEnabled(
                running
        );

        boolean editable =
                !running;

        sourceSpinner.setEnabled(
                editable
        );
        directionSpinner.setEnabled(
                editable
        );
        fieldSeek.setEnabled(
                editable
        );
        nsCheck.setEnabled(
                editable
        );
        aecCheck.setEnabled(
                editable
        );
        agcCheck.setEnabled(
                editable
        );
    }

    private void refreshDiagnostics() {
        StringBuilder text =
                new StringBuilder();

        text.append(
                "=== Android microphone inventory ===\n"
        );

        try {
            AudioManager manager =
                    (AudioManager) getSystemService(
                            Context.AUDIO_SERVICE
                    );

            String unprocessed =
                    manager.getProperty(
                            AudioManager.PROPERTY_SUPPORT_AUDIO_SOURCE_UNPROCESSED
                    );

            text.append(
                    "UNPROCESSED supported: "
            );
            text.append(
                    unprocessed
            );
            text.append(
                    "\n"
            );

            List<MicrophoneInfo> microphones =
                    manager.getMicrophones();

            text.append(
                    "Microphones reported: "
            );
            text.append(
                    microphones.size()
            );
            text.append(
                    "\n"
            );

            for (
                    int i = 0;
                    i < microphones.size();
                    i++
            ) {
                MicrophoneInfo mic =
                        microphones.get(
                                i
                        );

                text.append(
                        "\n#"
                );
                text.append(
                        i
                );
                text.append(
                        " id="
                );
                text.append(
                        mic.getId()
                );
                text.append(
                        " / type="
                );
                text.append(
                        mic.getType()
                );
                text.append(
                        " / "
                );
                text.append(
                        MicForegroundService.directionalityName(
                                mic.getDirectionality()
                        )
                );
                text.append(
                        "\n  location="
                );
                text.append(
                        mic.getLocation()
                );
                text.append(
                        " group="
                );
                text.append(
                        mic.getGroup()
                );
                text.append(
                        "/"
                );
                text.append(
                        mic.getIndexInTheGroup()
                );
                text.append(
                        "\n  sensitivity="
                );
                text.append(
                        mic.getSensitivity()
                );
                text.append(
                        " dBFS@94dBSPL"
                );
                text.append(
                        " min/maxSPL="
                );
                text.append(
                        mic.getMinSpl()
                );
                text.append(
                        "/"
                );
                text.append(
                        mic.getMaxSpl()
                );
                text.append(
                        "\n  freq-response points="
                );
                text.append(
                        mic.getFrequencyResponse()
                        .size()
                );
            }

        } catch (Exception exc) {
            text.append(
                    "Inventory query failed: "
            );
            text.append(
                    exc.getClass()
                    .getSimpleName()
            );
            text.append(
                    ": "
            );
            text.append(
                    exc.getMessage()
            );
            text.append(
                    "\n"
            );
        }

        SharedPreferences state =
                getSharedPreferences(
                        "state",
                        MODE_PRIVATE
                );

        text.append(
                "\n\n=== Active capture ===\n"
        );
        text.append(
                state.getString(
                        "active_mics",
                        "Native Mic을 시작하면 활성 microphone mapping이 표시됩니다."
                )
        );

        diagnosticsView.setText(
                text.toString()
        );
    }

    private void openBatterySettings() {
        try {
            Intent intent =
                    new Intent(
                            Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                            Uri.parse(
                                    "package:"
                                            + getPackageName()
                            )
                    );
            startActivity(
                    intent
            );

        } catch (Exception ignored) {
            Intent intent =
                    new Intent(
                            Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS
                    );
            startActivity(
                    intent
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults
    ) {
        super.onRequestPermissionsResult(
                requestCode,
                permissions,
                grantResults
        );

        if (
                requestCode
                        == REQUEST_AUDIO
        ) {
            boolean granted =
                    grantResults.length > 0
                            && grantResults[0]
                            == PackageManager.PERMISSION_GRANTED;

            if (granted) {
                requestNotificationPermissionIfNeeded();

                if (pendingAutoStart) {
                    startMicService();
                }
            } else {
                pendingAutoStart = false;
                statusView.setText(
                        "RECORD_AUDIO 권한이 필요합니다."
                );
            }
        }
    }
}
