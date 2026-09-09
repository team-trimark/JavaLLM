package team.trimark.javallm.type;

import java.util.Objects;

public class TokenMatrix {

    private int rows;
    private int columns;
    private float[] values;

    public TokenMatrix(Dimension size) {
        Objects.requireNonNull(size, "Size of a matrix cannot be null.");
    }
}
