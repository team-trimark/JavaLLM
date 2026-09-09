package team.trimark.javallm;

import team.trimark.javallm.tokenization.Detokenizer;
import team.trimark.javallm.tokenization.Tokenizer;
import team.trimark.javallm.type.TokenArray;

/**
 * The entry point of this application.
 */
public class Main {
    /**
     * Private constructor to prevent instantiation.
     */
    private Main() {}

    /**
     * Runs a fallback tokenize/detokenize round trip over a sample string.
     * @param args The command line arguments
     */
    public static void main(String[] args) {
        String testString = "Hello world, 안녕하세요!";
        System.out.println(testString);

        TokenArray tokenArray = Tokenizer.FALLBACK.apply(testString);
        System.out.println(tokenArray);

        String outputString = Detokenizer.FALLBACK.apply(tokenArray);
        System.out.println(outputString);
    }
}