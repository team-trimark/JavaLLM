package team.trimark.javallm.tokenization;

import team.trimark.javallm.type.TokenArray;

import java.util.function.Function;

/**
 * A tokenizer.
 */
public interface Tokenizer extends Function<String, TokenArray> {
    /**
     * The fallback tokenizer.
     */
    Tokenizer FALLBACK = new FallbackTokenizer();

    /**
     * Runs the tokenizer.
     * @param input The input string
     * @return The tokenized string
     */
    @Override
    TokenArray apply(String input);
}
