package local.vocalpitch.s24mic;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class MainActivity
        extends Activity {

    private static final int REQUEST_AUDIO = 1001;
    private static final int REQUEST_NOTIFICATIONS = 1002;

    private final Handler handler =
            new Handler(
                    Looper.getMainLooper()
            );

    private TextView statusView;
    private CheckBox nsCheck;
    private CheckBox aecCheck;
    private CheckBox agcCheck;
    private Button startButton;
    private Button stopButton;

    private boolean pendingAutoStart = false;

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
                "VPA S24 Native Mic"
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
                "Chrome가 아니라 Android microphone Foreground Service로 "
                        + "AudioRecord를 실행합니다.\n"
                        + "시작 후 상단에 지속 알림이 떠 있으면 화면을 끄거나 "
                        + "앱을 백그라운드로 보내도 PC 전송을 계속합니다.\n\n"
                        + "PC 앱이 먼저 실행되어 있고 adb reverse tcp:8791이 "
                        + "설정되어 있어야 합니다."
        );
        description.setTextSize(
                16
        );
        description.setPadding(
                0,
                20,
                0,
                20
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
                17
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

        nsCheck =
                new CheckBox(
                        this
                );
        nsCheck.setText(
                "Noise Suppression"
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

        Button batteryButton =
                new Button(
                        this
                );
        batteryButton.setText(
                "배터리 최적화 설정 열기"
        );
        batteryButton.setOnClickListener(
                view -> openBatterySettings()
        );
        root.addView(
                batteryButton,
                fullWidth()
        );

        TextView samsungNote =
                new TextView(
                        this
                );
        samsungNote.setText(
                "Galaxy에서 장시간 사용 중 서비스가 종료되면:\n"
                        + "설정 → 앱 → VPA S24 Mic → 배터리 → 제한 없음\n"
                        + "또는 절전 예외/자동 절전 제외 목록에 추가하세요."
        );
        samsungNote.setTextSize(
                14
        );
        samsungNote.setPadding(
                0,
                20,
                0,
                20
        );
        root.addView(
                samsungNote,
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
                .apply();
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
                        + "\n"
                        + text
        );

        startButton.setEnabled(
                !running
        );
        stopButton.setEnabled(
                running
        );

        nsCheck.setEnabled(
                !running
        );
        aecCheck.setEnabled(
                !running
        );
        agcCheck.setEnabled(
                !running
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
