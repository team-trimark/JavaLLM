package team.trimark.javallm.serve;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import team.trimark.javallm.llm.Checkpoint;
import team.trimark.javallm.llm.LanguageModel;
import team.trimark.javallm.llm.Vocabulary;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Serves a trained {@link LanguageModel} over the subset of the OpenAI API that a chat
 * client needs: listing models, and streaming or returning a completion.
 * <p>
 * Speaking a widely implemented protocol is what makes the model discoverable. Any tool
 * that can point at a base URL - OpenCode among them - can drive this without knowing
 * anything about the implementation behind it.
 * <p>
 * The model is a base continuation model trained on one book. It was never taught to
 * follow instructions or hold a conversation, so the handler feeds it the most recent user
 * message as raw text to continue, rather than a transcript of the exchange. Passing a
 * whole chat transcript, including a client's system prompt, would simply cause it to
 * continue that instead.
 */
public final class ChatServer {
    /**
     * The port served when none is given.
     */
    private static final int DEFAULT_PORT = 11435;

    /**
     * The number of tokens generated when a request does not say.
     */
    private static final int DEFAULT_MAX_TOKENS = 160;

    /**
     * The sampling temperature used when a request does not say.
     */
    private static final float DEFAULT_TEMPERATURE = 0.8f;

    /**
     * The number of highest scoring candidates sampled from.
     */
    private static final int DEFAULT_TOP_K = 20;

    /**
     * The greatest number of tokens any single request may generate.
     */
    private static final int TOKEN_LIMIT = 1024;

    /**
     * The model being served.
     */
    private final LanguageModel model;

    /**
     * The vocabulary the model predicts over.
     */
    private final Vocabulary vocabulary;

    /**
     * The name this model is advertised under.
     */
    private final String modelName;

    /**
     * The number of optimization steps the served weights were trained for.
     */
    private final int trainedSteps;

    /**
     * Serializes generation. A {@link LanguageModel} keeps mutable per-layer state across
     * a forward pass, so two concurrent requests would corrupt each other's activations.
     */
    private final ReentrantLock lock = new ReentrantLock();

    /**
     * The source of randomness used for sampling.
     */
    private final Random random = new Random();

    /**
     * Creates a server around the provided model.
     * @param model The model to serve
     * @param vocabulary The vocabulary the model predicts over
     * @param modelName The name to advertise the model under
     * @param trainedSteps The number of steps the weights were trained for
     */
    public ChatServer(LanguageModel model, Vocabulary vocabulary, String modelName, int trainedSteps) {
        this.model = model;
        this.vocabulary = vocabulary;
        this.modelName = modelName;
        this.trainedSteps = trainedSteps;
    }

    /**
     * Loads a checkpoint and serves it.
     * @param args The command line arguments
     * @throws IOException When the server cannot bind to its port
     */
    public static void main(String[] args) throws IOException {
        Path checkpoint = Paths.get(argument(args, "--checkpoint", "checkpoints/wealth-of-nations.ckpt"));
        int port = Integer.parseInt(argument(args, "--port", String.valueOf(DEFAULT_PORT)));
        String name = argument(args, "--name", "wealth-of-nations");

        System.out.println("loading " + checkpoint.toAbsolutePath());
        Checkpoint.Restored restored = Checkpoint.load(checkpoint).restore();

        ChatServer server = new ChatServer(restored.model(), restored.vocabulary(), name, restored.step());
        server.start(port);
    }

    /**
     * Binds the server and begins handling requests.
     * @param port The port to listen on
     * @throws IOException When the server cannot bind to its port
     */
    public void start(int port) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);

        server.createContext("/v1/models", this::handleModels);
        server.createContext("/v1/chat/completions", this::handleChatCompletions);
        server.createContext("/v1/completions", this::handleCompletions);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("model      : " + modelName);
        System.out.println("parameters : " + String.format("%,d", model.parameterCount()));
        System.out.println("vocabulary : " + vocabulary.size() + " tokens");
        System.out.println("trained    : " + trainedSteps + " steps");
        System.out.println("listening  : http://127.0.0.1:" + port + "/v1");
    }

    /**
     * Answers a request to list the available models.
     * @param exchange The exchange to answer
     * @throws IOException When the response cannot be written
     */
    private void handleModels(HttpExchange exchange) throws IOException {
        if (handledPreflight(exchange)) {
            return;
        }

        String body = "{\"object\":\"list\",\"data\":[{\"id\":" + Json.quote(modelName)
                + ",\"object\":\"model\",\"created\":" + created()
                + ",\"owned_by\":\"team.trimark.javallm\"}]}";

        respond(exchange, 200, "application/json", body);
    }

    /**
     * Answers a chat completion request.
     * @param exchange The exchange to answer
     * @throws IOException When the response cannot be written
     */
    private void handleChatCompletions(HttpExchange exchange) throws IOException {
        if (handledPreflight(exchange)) {
            return;
        }

        Map<String, Object> request;

        try {
            request = asObject(Json.parse(readBody(exchange)));
        } catch (RuntimeException e) {
            respond(exchange, 400, "application/json", errorBody("Malformed request: " + e.getMessage()));
            return;
        }

        String prompt = lastUserMessage(request);

        if (prompt == null || prompt.isBlank()) {
            respond(exchange, 400, "application/json", errorBody("No user message to continue."));
            return;
        }

        complete(exchange, request, prompt, true);
    }

    /**
     * Answers a legacy text completion request.
     * @param exchange The exchange to answer
     * @throws IOException When the response cannot be written
     */
    private void handleCompletions(HttpExchange exchange) throws IOException {
        if (handledPreflight(exchange)) {
            return;
        }

        Map<String, Object> request;

        try {
            request = asObject(Json.parse(readBody(exchange)));
        } catch (RuntimeException e) {
            respond(exchange, 400, "application/json", errorBody("Malformed request: " + e.getMessage()));
            return;
        }

        Object prompt = request.get("prompt");

        if (!(prompt instanceof String text) || text.isBlank()) {
            respond(exchange, 400, "application/json", errorBody("No prompt to continue."));
            return;
        }

        complete(exchange, request, text, false);
    }

    /**
     * Generates a continuation and writes it back, streaming when asked to.
     * @param exchange The exchange to answer
     * @param request The parsed request
     * @param prompt The text to continue
     * @param chat Whether to use the chat response shape
     * @throws IOException When the response cannot be written
     */
    private void complete(HttpExchange exchange, Map<String, Object> request, String prompt, boolean chat)
            throws IOException {
        int maxTokens = Math.min((int) number(request, "max_tokens", DEFAULT_MAX_TOKENS), TOKEN_LIMIT);
        float temperature = (float) number(request, "temperature", DEFAULT_TEMPERATURE);
        int topK = (int) number(request, "top_k", DEFAULT_TOP_K);
        boolean stream = Boolean.TRUE.equals(request.get("stream"));

        String id = (chat ? "chatcmpl-" : "cmpl-") + UUID.randomUUID();
        int promptTokens = vocabulary.encode(prompt).length;

        if (promptTokens == 0) {
            respond(exchange, 400, "application/json", errorBody("Prompt contains no tokens this model knows."));
            return;
        }

        if (!stream) {
            String continuation;

            lock.lock();

            try {
                continuation = generateContinuation(prompt, maxTokens, temperature, topK, null);
            } finally {
                lock.unlock();
            }

            respond(exchange, 200, "application/json",
                    chat ? chatBody(id, continuation, promptTokens) : textBody(id, continuation, promptTokens));
            return;
        }

        exchange.getResponseHeaders().add("Content-Type", "text/event-stream; charset=utf-8");
        exchange.getResponseHeaders().add("Cache-Control", "no-cache");
        exchange.getResponseHeaders().add("Connection", "keep-alive");
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(200, 0);

        try (OutputStream out = exchange.getResponseBody()) {
            lock.lock();

            try {
                send(out, chunkBody(id, "", "assistant", null, chat));
                generateContinuation(prompt, maxTokens, temperature, topK, piece -> {
                    try {
                        send(out, chunkBody(id, piece, null, null, chat));
                    } catch (IOException e) {
                        throw new java.io.UncheckedIOException(e);
                    }
                });
                send(out, chunkBody(id, "", null, "stop", chat));
                out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                out.flush();
            } finally {
                lock.unlock();
            }
        } catch (java.io.UncheckedIOException e) {
            // The client hung up part way through; nothing left to send.
        }
    }

    /**
     * Runs the model and returns only the newly generated text.
     * @param prompt The text to continue
     * @param maxTokens The number of tokens to generate
     * @param temperature The sampling temperature
     * @param topK The number of highest scoring candidates to sample from
     * @param onPiece Called with each fragment of text as it is produced, or {@code null}
     * @return The continuation, excluding the prompt
     */
    private String generateContinuation(String prompt, int maxTokens, float temperature, int topK,
                                        java.util.function.Consumer<String> onPiece) {
        StringBuilder produced = new StringBuilder();

        model.generate(prompt, maxTokens, temperature, topK, random, id -> {
            String piece = vocabulary.decode(new int[] {id});
            produced.append(piece);

            if (onPiece != null) {
                onPiece.accept(piece);
            }
        });

        return produced.toString();
    }

    /**
     * Writes one server-sent event.
     * @param out The stream to write to
     * @param json The payload of the event
     * @throws IOException When the event cannot be written
     */
    private static void send(OutputStream out, String json) throws IOException {
        out.write(("data: " + json + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    /**
     * Builds one streaming chunk.
     * @param id The identifier of this completion
     * @param content The text of this chunk
     * @param role The role to announce, or {@code null} to omit it
     * @param finishReason The reason generation stopped, or {@code null} if it has not
     * @param chat Whether to use the chat response shape
     * @return The encoded chunk
     */
    private String chunkBody(String id, String content, String role, String finishReason, boolean chat) {
        StringBuilder delta = new StringBuilder("{");

        if (role != null) {
            delta.append("\"role\":").append(Json.quote(role)).append(',');
        }

        delta.append("\"content\":").append(Json.quote(content)).append('}');

        String choice = chat
                ? "{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":" + quoteOrNull(finishReason) + "}"
                : "{\"index\":0,\"text\":" + Json.quote(content) + ",\"finish_reason\":" + quoteOrNull(finishReason) + "}";

        return "{\"id\":" + Json.quote(id)
                + ",\"object\":" + Json.quote(chat ? "chat.completion.chunk" : "text_completion")
                + ",\"created\":" + created()
                + ",\"model\":" + Json.quote(modelName)
                + ",\"choices\":[" + choice + "]}";
    }

    /**
     * Builds a complete chat response.
     * @param id The identifier of this completion
     * @param content The generated text
     * @param promptTokens The number of tokens in the prompt
     * @return The encoded response
     */
    private String chatBody(String id, String content, int promptTokens) {
        int completionTokens = vocabulary.encode(content).length;

        return "{\"id\":" + Json.quote(id)
                + ",\"object\":\"chat.completion\",\"created\":" + created()
                + ",\"model\":" + Json.quote(modelName)
                + ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" + Json.quote(content)
                + "},\"finish_reason\":\"stop\"}],\"usage\":" + usage(promptTokens, completionTokens) + "}";
    }

    /**
     * Builds a complete legacy text response.
     * @param id The identifier of this completion
     * @param content The generated text
     * @param promptTokens The number of tokens in the prompt
     * @return The encoded response
     */
    private String textBody(String id, String content, int promptTokens) {
        int completionTokens = vocabulary.encode(content).length;

        return "{\"id\":" + Json.quote(id)
                + ",\"object\":\"text_completion\",\"created\":" + created()
                + ",\"model\":" + Json.quote(modelName)
                + ",\"choices\":[{\"index\":0,\"text\":" + Json.quote(content)
                + ",\"finish_reason\":\"stop\"}],\"usage\":" + usage(promptTokens, completionTokens) + "}";
    }

    /**
     * Builds the token accounting block of a response.
     * @param promptTokens The number of tokens in the prompt
     * @param completionTokens The number of tokens generated
     * @return The encoded usage
     */
    private static String usage(int promptTokens, int completionTokens) {
        return "{\"prompt_tokens\":" + promptTokens
                + ",\"completion_tokens\":" + completionTokens
                + ",\"total_tokens\":" + (promptTokens + completionTokens) + "}";
    }

    /**
     * Returns the content of the most recent message with the user role.
     * @param request The parsed request
     * @return The content, or {@code null} when there is none
     */
    private static String lastUserMessage(Map<String, Object> request) {
        if (!(request.get("messages") instanceof List<?> messages)) {
            return null;
        }

        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i) instanceof Map<?, ?> message
                    && "user".equals(message.get("role"))
                    && message.get("content") instanceof String content) {
                return content;
            }
        }

        return null;
    }

    /**
     * Reads a numeric field of the request, falling back when it is absent or null.
     * @param request The parsed request
     * @param name The field to read
     * @param fallback The value to use when the field is absent
     * @return The value
     */
    private static double number(Map<String, Object> request, String name, double fallback) {
        return request.get(name) instanceof Double value ? value : fallback;
    }

    /**
     * Casts a parsed value to an object, rejecting anything else.
     * @param value The parsed value
     * @return The object
     * @throws IllegalArgumentException When the value is not an object
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> asObject(Object value) {
        if (!(value instanceof Map)) {
            throw new IllegalArgumentException("Expected a JSON object.");
        }

        return (Map<String, Object>) value;
    }

    /**
     * Quotes a string, or renders {@code null} when it is absent.
     * @param value The value to render
     * @return The rendered value
     */
    private static String quoteOrNull(String value) {
        return value == null ? "null" : Json.quote(value);
    }

    /**
     * Returns the current time in whole seconds since the epoch.
     * @return The timestamp
     */
    private static long created() {
        return System.currentTimeMillis() / 1000L;
    }

    /**
     * Builds an error response body.
     * @param message The message to report
     * @return The encoded error
     */
    private static String errorBody(String message) {
        return "{\"error\":{\"message\":" + Json.quote(message) + ",\"type\":\"invalid_request_error\"}}";
    }

    /**
     * Answers a cross-origin preflight request, if that is what this is.
     * @param exchange The exchange to inspect
     * @return {@code true} when the request has been fully answered
     * @throws IOException When the response cannot be written
     */
    private static boolean handledPreflight(HttpExchange exchange) throws IOException {
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");

        if (!"OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
            return false;
        }

        exchange.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type, Authorization");
        exchange.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        exchange.sendResponseHeaders(204, -1);
        exchange.close();

        return true;
    }

    /**
     * Reads the body of a request as text.
     * @param exchange The exchange to read
     * @return The body
     * @throws IOException When the body cannot be read
     */
    private static String readBody(HttpExchange exchange) throws IOException {
        return new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    /**
     * Writes a complete response.
     * @param exchange The exchange to answer
     * @param status The status code
     * @param contentType The media type of the body
     * @param body The body
     * @throws IOException When the response cannot be written
     */
    private static void respond(HttpExchange exchange, int status, String contentType, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);

        exchange.getResponseHeaders().add("Content-Type", contentType + "; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);

        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    /**
     * Returns the value following the provided option, or a fallback when it is absent.
     * @param args The command line arguments
     * @param name The option to look for
     * @param fallback The value to use when the option is absent
     * @return The value
     */
    private static String argument(String[] args, String name, String fallback) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return args[i + 1];
            }
        }

        return fallback;
    }
}
