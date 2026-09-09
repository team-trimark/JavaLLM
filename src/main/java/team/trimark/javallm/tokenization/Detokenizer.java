package team.trimark.javallm.tokenization;

import team.trimark.javallm.type.TokenArray;

import java.util.function.Function;

/**
 * A detokenizer.
 */
public interface Detokenizer extends Function<TokenArray, String> {
    /**
     * The fallback detokenizer.
     */
    Detokenizer FALLBACK = new FallbackDetokenizer();

    /**
     * Runs the detokenizer.
     * @param tokens The tokens
     * @return The string
     */
    @Override
    String apply(TokenArray tokens);
}
