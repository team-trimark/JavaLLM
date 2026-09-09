package team.trimark.javallm.type;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;

/**
 * A two-dimensional, fixed-size grid of tokens. Values are stored flat in column-major order.
 */
public class TokenMatrix {

    /**
     * The number of rows of this matrix.
     */
    private final int rows;

    /**
     * The number of columns of this matrix.
     */
    private final int columns;

    /**
     * The flat backing values of this matrix.
     */
    private final float[] values;

    /**
     * Creates a new matrix of the provided dimensions, with every value initialized to {@code 0}.
     * @param rows The number of rows
     * @param columns The number of columns
     * @throws IllegalArgumentException When either dimension is negative, or their product overflows
     */
    public TokenMatrix(int rows, int columns) {
        if (rows < 0 || columns < 0 || (long) rows * columns > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Dimensions are invalid or too large: " + Dimension.rowCols(rows, columns));
        }

        this.rows = rows;
        this.columns = columns;
        this.values = new float[rows * columns];

    }

    /**
     * Creates a new matrix of the provided dimensions, with every value initialized to {@code 0}.
     * @param size The dimensions of the new matrix
     * @throws IllegalArgumentException When either dimension is negative, or their product overflows
     * @throws NullPointerException When the provided size is {@code null}
     */
    public TokenMatrix(Dimension size) {
        Objects.requireNonNull(size, "Size of a matrix cannot be null.");

        this.rows = size.rows();
        this.columns = size.columns();

        if (rows < 0 || columns < 0 || (long) rows * columns > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Dimensions are invalid or too large: " + size);
        }

        this.values = new float[rows * columns];
    }

    /**
     * Creates a new matrix from another one. Produces a deep copy, where any change from the source
     * will not be reflected in the new instance.
     * @param m The source to copy from
     * @throws NullPointerException When the source is {@code null}
     */
    public TokenMatrix(TokenMatrix m) {
        Objects.requireNonNull(m, "Cannot copy from null matrix.");

        this.rows = m.rows;
        this.columns = m.columns;
        this.values = Arrays.copyOf(m.values, rows * columns);
    }

    /**
     * Returns the number of rows of this matrix.
     * @return The number of rows
     */
    public int rows() {
        return rows;
    }

    /**
     * Returns the number of columns of this matrix.
     * @return The number of columns
     */
    public int columns() {
        return columns;
    }

    /**
     * Converts the row-column pair held by the provided dimension into a flat index.
     * @param dim The row-column pair to convert
     * @return The corresponding flat index
     */
    private int dimsToFlat(Dimension dim) {
        return dim.columns() * rows + dim.rows();
    }

    /**
     * Converts a row-column pair into a flat index.
     * @param r The row
     * @param c The column
     * @return The corresponding flat index
     */
    private int rowColToFlat(int r, int c) {
        return c * rows + r;
    }

    /**
     * Converts a flat index into a row-column pair.
     * @param i The flat index to convert
     * @return The corresponding row-column pair
     */
    private Dimension flatToRowCol(int i) {
        int r = i % rows;
        int c = i / rows;
        return Dimension.rowCols(r, c);
    }

    /**
     * Returns the value at the provided row and column.
     * @param r The row to get
     * @param c The column to get
     * @return The value
     * @throws IndexOutOfBoundsException When either coordinate is out of bounds
     */
    public float get(int r, int c) throws IndexOutOfBoundsException {
        if (r >= rows || c >= columns || r < 0 || c < 0) {
            throw new IndexOutOfBoundsException("Index out of bounds for matrix " + rows + " * " + columns);
        }

        int i = rowColToFlat(r, c);
        return values[i];
    }

    /**
     * Sets the value at the provided row and column.
     * @param r The row to set
     * @param c The column to set
     * @param v The value to set to
     * @throws IndexOutOfBoundsException When either coordinate is out of bounds
     */
    public void set(int r, int c, float v) throws IndexOutOfBoundsException {
        if (r >= rows || c >= columns || r < 0 || c < 0) {
            throw new IndexOutOfBoundsException("Index out of bounds for matrix " + rows + " * " + columns);
        }

        int i = rowColToFlat(r, c);
        values[i] = v;
    }

    /**
     * Checks for equality with another object.
     * @param obj The object to compare to
     * @return {@code true} if the other object is a matrix of equal dimensions and contents
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof TokenMatrix tm)) return false;
        if (rows != tm.rows || columns != tm.columns) return false;
        return Arrays.equals(values, tm.values);
    }

    /**
     * Returns the hash code of this matrix.
     * @return The hash code of this matrix
     */
    @Override
    public int hashCode() {
        return Objects.hash(rows, columns, Arrays.hashCode(values));
    }

    /**
     * Returns the contents of this matrix as a two-dimensional array, indexed by row then column.
     * @return The contents of this matrix
     */
    public float[][] toFloatArray() {
        float[][] result = new float[rows][columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[r][c] = values[rowColToFlat(r, c)];
            }
        }

        return result;
    }

    /**
     * Returns the contents of this matrix as a two-dimensional array of strings, indexed by row then
     * column, using {@link Object#toString()} to format each value.
     * @return The contents of this matrix
     */
    public String[][] toStringArray() {
        return toStringArray(Object::toString);
    }

    /**
     * Returns the contents of this matrix as a two-dimensional array of strings, indexed by row then
     * column.
     * @param formatter The formatter to convert each value with
     * @return The contents of this matrix
     */
    public String[][] toStringArray(Function<Float, String> formatter) {
        String[][] result = new String[rows][columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[r][c] = formatter.apply(values[rowColToFlat(r, c)]);
            }
        }

        return result;
    }

    /**
     * Serializes this matrix into a string.
     * @return The string representation of this matrix
     */
    @Override
    public String toString() {
        return "TokenMatrix{" +
                "rows=" + rows +
                ", columns=" + columns +
                ", values=" + Arrays.toString(values) +
                '}';
    }
}
