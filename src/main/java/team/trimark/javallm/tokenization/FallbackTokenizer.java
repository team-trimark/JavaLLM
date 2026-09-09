package team.trimark.javallm.tokenization;

import team.trimark.javallm.type.TokenArray;

/**
 * Fallback tokenizer which splits every letter but blank space.
 */
public class FallbackTokenizer implements Tokenizer {
    /**
     * Creates a new fallback tokenizer.
     */
    public FallbackTokenizer() {}

    /**
     * Splits the input into one token per code point, dropping blank spaces.
     * @param input The input string
     * @return The tokenized string
     */
    @Override
    public TokenArray apply(String input) {
        int maxTokenCount = input.length();
        TokenArray tokens = new TokenArray(maxTokenCount);

        for (int i = 0; i < maxTokenCount; ) {
            int token = input.codePointAt(i);
            i += Character.charCount(token);

            if (token == ' ') continue;
            tokens.append(Float.intBitsToFloat(token));
        }

        tokens.trim();
        return tokens;
    }
}
