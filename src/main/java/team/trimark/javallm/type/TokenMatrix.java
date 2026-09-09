package team.trimark.javallm.type;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;

public class TokenMatrix {

    private final int rows;
    private final int columns;
    private final float[] values;

    public TokenMatrix(int rows, int columns) {
        if (rows < 0 || columns < 0 || rows * columns < 0) {
            throw new IllegalArgumentException("Dimensions are invalid or too large: " + Dimension.rowCols(rows, columns));
        }

        this.rows = rows;
        this.columns = columns;
        this.values = new float[rows * columns];

    }

    public TokenMatrix(Dimension size) {
        Objects.requireNonNull(size, "Size of a matrix cannot be null.");

        this.rows = size.rows();
        this.columns = size.columns();

        if (rows < 0 || columns < 0 || size.area() < 0) {
            throw new IllegalArgumentException("Dimensions are invalid or too large: " + size);
        }

        this.values = new float[size.area()];
    }

    public TokenMatrix(TokenMatrix m) {
        Objects.requireNonNull(m, "Cannot copy from null matrix.");

        this.rows = m.rows;
        this.columns = m.columns;
        this.values = Arrays.copyOf(m.values, rows * columns);
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    private int dimsToFlat(Dimension dim) {
        return dim.columns() * rows + dim.rows();
    }

    private int rowColToFlat(int r, int c) {
        return c * rows + r;
    }

    private Dimension flatToRowCol(int i) {
        int r = i % rows;
        int c = i / rows;
        return Dimension.rowCols(r, c);
    }

    public float get(int r, int c) throws IndexOutOfBoundsException {
        if (r >= rows || c >= columns || r < 0 || c < 0) {
            throw new IndexOutOfBoundsException("Index out of bounds for matrix " + rows + " * " + columns);
        }

        int i = rowColToFlat(r, c);
        return values[i];
    }

    public void set(int r, int c, float v) throws IndexOutOfBoundsException {
        if (r >= rows || c >= columns || r < 0 || c < 0) {
            throw new IndexOutOfBoundsException("Index out of bounds for matrix " + rows + " * " + columns);
        }

        int i = rowColToFlat(r, c);
        values[i] = v;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof TokenMatrix tm)) return false;
        if (rows != tm.rows || columns != tm.columns) return false;
        return Arrays.equals(values, tm.values);
    }

    @Override
    public int hashCode() {
        return Objects.hash(rows, columns, Arrays.hashCode(values));
    }

    public float[][] toDoubleArray() {
        float[][] result = new float[rows][columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[r][c] = values[rowColToFlat(r, c)];
            }
        }

        return result;
    }

    public String[][] toStringArray() {
        return toStringArray(Object::toString);
    }

    public String[][] toStringArray(Function<Float, String> formatter) {
        String[][] result = new String[rows][columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[r][c] = formatter.apply(values[rowColToFlat(r, c)]);
            }
        }

        return result;
    }

    @Override
    public String toString() {
        return "TokenMatrix{" +
                "rows=" + rows +
                ", columns=" + columns +
                ", values=" + Arrays.toString(values) +
                '}';
    }
}
