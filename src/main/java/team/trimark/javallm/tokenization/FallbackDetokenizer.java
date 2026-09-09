package team.trimark.javallm.tokenization;

import team.trimark.javallm.type.TokenArray;

/**
 * Detokenizer which interprets raw.
 */
public class FallbackDetokenizer implements Detokenizer {
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
