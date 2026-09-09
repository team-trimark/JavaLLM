package team.trimark.javallm.tokenization;

import team.trimark.javallm.type.TokenArray;

/**
 * Fallback tokenizer which splits every letter but blank space.
 */
public class FallbackTokenizer implements Tokenizer {
    @Override
    public TokenArray apply(String input) {
        int maxTokenCount = input.length();
        TokenArray tokens = new TokenArray(maxTokenCount);

        for (int i = 0; i < maxTokenCount; i++) {
            char token = input.charAt(i);
            if (token == ' ') continue;
            tokens.append(Float.intBitsToFloat(input.codePointAt(i)));
        }

        tokens.trim();
        return tokens;
    }
}
