package team.trimark.javallm;

import team.trimark.javallm.type.TokenArray;

import java.util.Random;

public class Main {
    public static void main(String[] args) {
        TokenArray array = new TokenArray();
        Random rand = new Random();

        for (int i = 0; i < 10; i++) {
            array.append(rand.nextFloat());
        }

        System.out.println(array);
    }
}