package team.trimark.javallm;

import team.trimark.javallm.tokenization.Detokenizer;
import team.trimark.javallm.tokenization.Tokenizer;
import team.trimark.javallm.type.TokenArray;

import java.util.Random;

public class Main {
    public static void main(String[] args) {
        String testString = "Hello world, 안녕하세요!";
        System.out.println(testString);

        TokenArray tokenArray = Tokenizer.FALLBACK.apply(testString);
        System.out.println(tokenArray);

        String outputString = Detokenizer.FALLBACK.apply(tokenArray);
        System.out.println(outputString);
    }
}