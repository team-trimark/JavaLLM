package team.trimark.javallm.llm;

import team.trimark.javallm.tokenization.Tokenizer;
import team.trimark.javallm.type.TokenArray;

/**
 * A lossless tokenizer which emits exactly one token per Unicode code point, including whitespace.
 * Unlike {@link team.trimark.javallm.tokenization.FallbackTokenizer} it discards nothing, so a
 * tokenize then detokenize round trip always reproduces the original string.
 */
public class CodePointTokenizer implements Tokenizer {
    /**
     * Creates a new code point tokenizer.
     */
    public CodePointTokenizer() {}

    /**
     * Splits the input into one token per code point.
     * @param input The input string
     * @return The tokenized string
     */
    @Override
    public TokenArray apply(String input) {
        int length = input.length();
        TokenArray tokens = new TokenArray(length);

        for (int i = 0; i < length; ) {
            int codePoint = input.codePointAt(i);
            i += Character.charCount(codePoint);

            tokens.append(Float.intBitsToFloat(codePoint));
        }

        tokens.trim();
        return tokens;
    }
}
