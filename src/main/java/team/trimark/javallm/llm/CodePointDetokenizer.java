package team.trimark.javallm.llm;

import team.trimark.javallm.tokenization.Detokenizer;
import team.trimark.javallm.type.TokenArray;

/**
 * The inverse of {@link CodePointTokenizer}, reassembling a string from one token per code point.
 */
public class CodePointDetokenizer implements Detokenizer {
    /**
     * Creates a new code point detokenizer.
     */
    public CodePointDetokenizer() {}

    /**
     * Concatenates every token back into a string, dropping any token which is not a valid code point.
     * @param tokens The tokens
     * @return The string
     */
    @Override
    public String apply(TokenArray tokens) {
        StringBuilder builder = new StringBuilder();

        for (float token : tokens) {
            try {
                builder.append(Character.toChars(Float.floatToRawIntBits(token)));
            } catch (IllegalArgumentException ignored) {
                // Not a valid code point, so there is nothing to append.
            }
        }

        return builder.toString();
    }
}
