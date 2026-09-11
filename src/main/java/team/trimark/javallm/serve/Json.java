package team.trimark.javallm.serve;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A minimal JSON reader and writer, sufficient for the request and response shapes of a
 * chat completions API. This project has no dependencies, so rather than pull one in for
 * a handful of object literals, the format is small enough to implement outright.
 * <p>
 * Parsed values map onto {@link Map}, {@link List}, {@link String}, {@link Double},
 * {@link Boolean} and {@code null}. Objects preserve their insertion order.
 */
public final class Json {
    /**
     * The text being parsed.
     */
    private final String text;

    /**
     * The offset of the next character to read.
     */
    private int index;

    /**
     * Creates a parser over the provided text.
     * @param text The text to parse
     */
    private Json(String text) {
        this.text = text;
    }

    /**
     * Parses a single JSON value from the provided text.
     * @param text The text to parse
     * @return The parsed value
     * @throws NullPointerException When the text is {@code null}
     * @throws IllegalArgumentException When the text is not well-formed JSON
     */
    public static Object parse(String text) {
        Objects.requireNonNull(text, "JSON text cannot be null.");

        Json parser = new Json(text);
        parser.skipWhitespace();
        Object value = parser.readValue();
        parser.skipWhitespace();

        if (parser.index < text.length()) {
            throw new IllegalArgumentException("Trailing content at offset " + parser.index + ".");
        }

        return value;
    }

    /**
     * Reads a value of any type at the current position.
     * @return The value
     * @throws IllegalArgumentException When no valid value begins here
     */
    private Object readValue() {
        if (index >= text.length()) {
            throw new IllegalArgumentException("Unexpected end of input.");
        }

        char c = text.charAt(index);

        return switch (c) {
            case '{' -> readObject();
            case '[' -> readArray();
            case '"' -> readString();
            case 't' -> readLiteral("true", Boolean.TRUE);
            case 'f' -> readLiteral("false", Boolean.FALSE);
            case 'n' -> readLiteral("null", null);
            default -> readNumber();
        };
    }

    /**
     * Reads an object at the current position.
     * @return The object, as a map in insertion order
     * @throws IllegalArgumentException When the object is malformed
     */
    private Map<String, Object> readObject() {
        Map<String, Object> object = new LinkedHashMap<>();
        index++;
        skipWhitespace();

        if (peek() == '}') {
            index++;
            return object;
        }

        while (true) {
            skipWhitespace();
            String key = readString();
            skipWhitespace();
            expect(':');
            skipWhitespace();
            object.put(key, readValue());
            skipWhitespace();

            char c = peek();
            index++;

            if (c == '}') {
                return object;
            }

            if (c != ',') {
                throw new IllegalArgumentException("Expected , or } at offset " + (index - 1) + ".");
            }
        }
    }

    /**
     * Reads an array at the current position.
     * @return The array, as a list
     * @throws IllegalArgumentException When the array is malformed
     */
    private List<Object> readArray() {
        List<Object> array = new ArrayList<>();
        index++;
        skipWhitespace();

        if (peek() == ']') {
            index++;
            return array;
        }

        while (true) {
            skipWhitespace();
            array.add(readValue());
            skipWhitespace();

            char c = peek();
            index++;

            if (c == ']') {
                return array;
            }

            if (c != ',') {
                throw new IllegalArgumentException("Expected , or ] at offset " + (index - 1) + ".");
            }
        }
    }

    /**
     * Reads a quoted string at the current position, resolving escape sequences.
     * @return The string
     * @throws IllegalArgumentException When the string is malformed
     */
    private String readString() {
        expect('"');
        StringBuilder builder = new StringBuilder();

        while (true) {
            if (index >= text.length()) {
                throw new IllegalArgumentException("Unterminated string.");
            }

            char c = text.charAt(index++);

            if (c == '"') {
                return builder.toString();
            }

            if (c != '\\') {
                builder.append(c);
                continue;
            }

            char escape = text.charAt(index++);

            switch (escape) {
                case '"', '\\', '/' -> builder.append(escape);
                case 'b' -> builder.append('\b');
                case 'f' -> builder.append('\f');
                case 'n' -> builder.append('\n');
                case 'r' -> builder.append('\r');
                case 't' -> builder.append('\t');
                case 'u' -> {
                    builder.append((char) Integer.parseInt(text.substring(index, index + 4), 16));
                    index += 4;
                }
                default -> throw new IllegalArgumentException("Invalid escape \\" + escape + ".");
            }
        }
    }

    /**
     * Reads a number at the current position.
     * @return The number
     * @throws IllegalArgumentException When the number is malformed
     */
    private Double readNumber() {
        int start = index;

        while (index < text.length() && "+-.eE0123456789".indexOf(text.charAt(index)) >= 0) {
            index++;
        }

        try {
            return Double.valueOf(text.substring(start, index));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid number at offset " + start + ".", e);
        }
    }

    /**
     * Reads a fixed literal at the current position.
     * @param literal The text to require
     * @param value The value the literal denotes
     * @return The value
     * @throws IllegalArgumentException When the literal is not present
     */
    private Object readLiteral(String literal, Object value) {
        if (!text.startsWith(literal, index)) {
            throw new IllegalArgumentException("Invalid literal at offset " + index + ".");
        }

        index += literal.length();
        return value;
    }

    /**
     * Returns the character at the current position without consuming it.
     * @return The character
     * @throws IllegalArgumentException When the input has ended
     */
    private char peek() {
        if (index >= text.length()) {
            throw new IllegalArgumentException("Unexpected end of input.");
        }

        return text.charAt(index);
    }

    /**
     * Consumes the current character, requiring it to be the one provided.
     * @param c The required character
     * @throws IllegalArgumentException When a different character is present
     */
    private void expect(char c) {
        if (peek() != c) {
            throw new IllegalArgumentException("Expected " + c + " at offset " + index + ".");
        }

        index++;
    }

    /**
     * Advances past any whitespace at the current position.
     */
    private void skipWhitespace() {
        while (index < text.length() && Character.isWhitespace(text.charAt(index))) {
            index++;
        }
    }

    /**
     * Quotes and escapes a string for inclusion in JSON output.
     * @param value The string to escape
     * @return The quoted string, including its surrounding quotation marks
     */
    public static String quote(String value) {
        StringBuilder builder = new StringBuilder("\"");

        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);

            switch (c) {
                case '"' -> builder.append("\\\"");
                case '\\' -> builder.append("\\\\");
                case '\n' -> builder.append("\\n");
                case '\r' -> builder.append("\\r");
                case '\t' -> builder.append("\\t");
                case '\b' -> builder.append("\\b");
                case '\f' -> builder.append("\\f");
                default -> {
                    if (c < 0x20) {
                        builder.append(String.format("\\u%04x", (int) c));
                    } else {
                        builder.append(c);
                    }
                }
            }
        }

        return builder.append('"').toString();
    }
}
