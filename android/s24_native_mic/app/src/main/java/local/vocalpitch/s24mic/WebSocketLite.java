package local.vocalpitch.s24mic;

import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Locale;

final class WebSocketLite {
    private static final String GUID =
            "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

    private final SecureRandom random = new SecureRandom();
    private final Object sendLock = new Object();

    private Socket socket;
    private InputStream input;
    private OutputStream output;
    private Thread readerThread;

    private volatile boolean open = false;
    private volatile String lastError = "";

    boolean isOpen() {
        return open
                && socket != null
                && socket.isConnected()
                && !socket.isClosed();
    }

    String getLastError() {
        return lastError;
    }

    void connect(
            String host,
            int port,
            String path
    ) throws Exception {
        close();

        Socket next = new Socket();
        next.setTcpNoDelay(true);
        next.setKeepAlive(true);
        next.connect(
                new InetSocketAddress(
                        host,
                        port
                ),
                5000
        );

        InputStream nextInput = next.getInputStream();
        OutputStream nextOutput = next.getOutputStream();

        byte[] keyBytes = new byte[16];
        random.nextBytes(keyBytes);
        String key = Base64.encodeToString(
                keyBytes,
                Base64.NO_WRAP
        );

        String request =
                "GET " + path + " HTTP/1.1\r\n"
                        + "Host: " + host + ":" + port + "\r\n"
                        + "Upgrade: websocket\r\n"
                        + "Connection: Upgrade\r\n"
                        + "Sec-WebSocket-Key: " + key + "\r\n"
                        + "Sec-WebSocket-Version: 13\r\n"
                        + "\r\n";

        nextOutput.write(
                request.getBytes(
                        StandardCharsets.US_ASCII
                )
        );
        nextOutput.flush();

        String response = readHttpHeaders(
                nextInput
        );

        String firstLine = response
                .split(
                        "\r\n",
                        2
                )[0];

        if (!firstLine.contains(" 101 ")) {
            next.close();
            throw new IOException(
                    "WebSocket handshake failed: "
                            + firstLine
            );
        }

        String expectedAccept = expectedAccept(
                key
        );

        String actualAccept = headerValue(
                response,
                "Sec-WebSocket-Accept"
        );

        if (
                actualAccept == null
                        || !expectedAccept.equals(
                        actualAccept.trim()
                )
        ) {
            next.close();
            throw new IOException(
                    "Invalid Sec-WebSocket-Accept"
            );
        }

        socket = next;
        input = nextInput;
        output = nextOutput;
        open = true;
        lastError = "";

        readerThread = new Thread(
                this::readerLoop,
                "S24Mic-WebSocketReader"
        );
        readerThread.setDaemon(true);
        readerThread.start();
    }

    void sendText(
            String text
    ) throws IOException {
        sendFrame(
                0x1,
                text.getBytes(
                        StandardCharsets.UTF_8
                )
        );
    }

    void sendBinary(
            byte[] payload
    ) throws IOException {
        sendFrame(
                0x2,
                payload
        );
    }

    private void sendPong(
            byte[] payload
    ) throws IOException {
        sendFrame(
                0xA,
                payload
        );
    }

    private void sendFrame(
            int opcode,
            byte[] payload
    ) throws IOException {
        if (!isOpen()) {
            throw new IOException(
                    "WebSocket not open"
            );
        }

        byte[] mask = new byte[4];
        random.nextBytes(mask);

        synchronized (sendLock) {
            OutputStream out = output;

            if (out == null) {
                throw new IOException(
                        "WebSocket output closed"
                );
            }

            out.write(
                    0x80
                            | (
                            opcode
                                    & 0x0F
                    )
            );

            int length = payload.length;

            if (length < 126) {
                out.write(
                        0x80
                                | length
                );
            } else if (length <= 0xFFFF) {
                out.write(
                        0x80
                                | 126
                );
                out.write(
                        (
                                length
                                        >>> 8
                        )
                                & 0xFF
                );
                out.write(
                        length
                                & 0xFF
                );
            } else {
                out.write(
                        0x80
                                | 127
                );

                long value = length;

                for (
                        int shift = 56;
                        shift >= 0;
                        shift -= 8
                ) {
                    out.write(
                            (int) (
                                    value
                                            >>> shift
                            )
                                    & 0xFF
                    );
                }
            }

            out.write(
                    mask
            );

            byte[] masked = new byte[
                    payload.length
                    ];

            for (
                    int i = 0;
                    i < payload.length;
                    i++
            ) {
                masked[i] = (byte) (
                        payload[i]
                                ^ mask[
                                i
                                        & 3
                                ]
                );
            }

            out.write(
                    masked
            );
            out.flush();
        }
    }

    private void readerLoop() {
        try {
            while (isOpen()) {
                int first = readRequired(
                        input
                );
                int second = readRequired(
                        input
                );

                int opcode = first
                        & 0x0F;
                boolean masked = (
                        second
                                & 0x80
                ) != 0;
                long length = second
                        & 0x7F;

                if (length == 126) {
                    length =
                            (
                                    readRequired(
                                            input
                                    )
                                            << 8
                            )
                                    | readRequired(
                                    input
                            );
                } else if (length == 127) {
                    long value = 0;

                    for (
                            int i = 0;
                            i < 8;
                            i++
                    ) {
                        value =
                                (
                                        value
                                                << 8
                                )
                                        | readRequired(
                                        input
                                );
                    }

                    length = value;
                }

                if (
                        length < 0
                                || length > 16L
                                * 1024L
                                * 1024L
                ) {
                    throw new IOException(
                            "Invalid frame length: "
                                    + length
                    );
                }

                byte[] mask = null;

                if (masked) {
                    mask = readExact(
                            input,
                            4
                    );
                }

                byte[] payload = readExact(
                        input,
                        (int) length
                );

                if (masked) {
                    for (
                            int i = 0;
                            i < payload.length;
                            i++
                    ) {
                        payload[i] = (byte) (
                                payload[i]
                                        ^ mask[
                                        i
                                                & 3
                                        ]
                        );
                    }
                }

                if (opcode == 0x9) {
                    sendPong(
                            payload
                    );
                } else if (opcode == 0x8) {
                    break;
                }
            }

        } catch (Exception exc) {
            lastError =
                    exc.getClass()
                            .getSimpleName()
                            + ": "
                            + exc.getMessage();

        } finally {
            closeInternal();
        }
    }

    void close() {
        if (isOpen()) {
            try {
                sendFrame(
                        0x8,
                        new byte[0]
                );
            } catch (Exception ignored) {
            }
        }

        closeInternal();
    }

    private void closeInternal() {
        open = false;

        Socket old = socket;
        socket = null;
        input = null;
        output = null;

        if (old != null) {
            try {
                old.close();
            } catch (Exception ignored) {
            }
        }
    }

    private static int readRequired(
            InputStream input
    ) throws IOException {
        int value = input.read();

        if (value < 0) {
            throw new EOFException();
        }

        return value;
    }

    private static byte[] readExact(
            InputStream input,
            int length
    ) throws IOException {
        byte[] output = new byte[
                length
                ];
        int offset = 0;

        while (offset < length) {
            int count = input.read(
                    output,
                    offset,
                    length
                            - offset
            );

            if (count < 0) {
                throw new EOFException();
            }

            offset += count;
        }

        return output;
    }

    private static String readHttpHeaders(
            InputStream input
    ) throws IOException {
        ByteArrayOutputStream buffer =
                new ByteArrayOutputStream();

        int state = 0;

        while (buffer.size() < 16384) {
            int value = input.read();

            if (value < 0) {
                throw new EOFException(
                        "Handshake EOF"
                );
            }

            buffer.write(
                    value
            );

            if (
                    state == 0
                            && value == '\r'
            ) {
                state = 1;
            } else if (
                    state == 1
                            && value == '\n'
            ) {
                state = 2;
            } else if (
                    state == 2
                            && value == '\r'
            ) {
                state = 3;
            } else if (
                    state == 3
                            && value == '\n'
            ) {
                break;
            } else {
                state = 0;
            }
        }

        return buffer.toString(
                StandardCharsets.US_ASCII.name()
        );
    }

    private static String expectedAccept(
            String key
    ) throws Exception {
        MessageDigest sha1 = MessageDigest.getInstance(
                "SHA-1"
        );

        byte[] digest = sha1.digest(
                (
                        key
                                + GUID
                ).getBytes(
                        StandardCharsets.US_ASCII
                )
        );

        return Base64.encodeToString(
                digest,
                Base64.NO_WRAP
        );
    }

    private static String headerValue(
            String headers,
            String name
    ) {
        String prefix = name
                .toLowerCase(
                        Locale.ROOT
                )
                + ":";

        for (
                String line :
                headers.split(
                        "\r\n"
                )
        ) {
            String lowered = line
                    .toLowerCase(
                            Locale.ROOT
                    );

            if (
                    lowered.startsWith(
                            prefix
                    )
            ) {
                int colon = line.indexOf(
                        ':'
                );

                if (colon >= 0) {
                    return line.substring(
                            colon
                                    + 1
                    ).trim();
                }
            }
        }

        return null;
    }
}
