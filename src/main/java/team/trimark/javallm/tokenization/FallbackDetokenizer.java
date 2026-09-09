package team.trimark.javallm.tokenization;

import team.trimark.javallm.type.TokenArray;

/**
 * Detokenizer which interprets raw.
 */
public class FallbackDetokenizer implements Detokenizer {
    /**
     * Creates a new fallback detokenizer.
     */
    public FallbackDetokenizer() {}

    /**
     * Concatenates every token back into a string, dropping any token which is not a valid code point.
     * @param tokens The tokens
     * @return The string
     */
    @Override
    public String apply(TokenArray tokens) {
        StringBuilder sb = new StringBuilder();

        for (float token : tokens) {
            char[] text;
            try {
                text = Character.toChars(Float.floatToRawIntBits(token));
            } catch (RuntimeException e) {
                text = new char[0];
            }
            sb.append(text);
        }

        return sb.toString();
    }
}
