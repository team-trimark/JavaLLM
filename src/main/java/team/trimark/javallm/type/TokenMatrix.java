package team.trimark.javallm.type;

import java.util.Objects;
import java.util.function.Function;

/**
 * A two-dimensional, fixed-size grid of floating point values.
 * <p>
 * This is a contract, not a layout. An implementation is free to store its values however it
 * likes - densely, sparsely, or as a view over another matrix - so long as it answers
 * {@link #get(int, int)} and {@link #set(int, int, float)} consistently with the shape it
 * reports. Callers that only read and write individual values need nothing more than this.
 * <p>
 * Callers that move bulk data - the product kernels in particular - want more than that: they
 * want to walk memory contiguously, which per-element accessors cannot express. Those callers
 * should test for {@link DenseTokenMatrix} and take its backing array when it is available,
 * falling back to {@code get}/{@code set} when it is not. Keeping that capability on the
 * implementation rather than in this interface is deliberate: exposing a mutable array from
 * the contract would oblige every future implementation to have one.
 *
 * @see DenseTokenMatrix The default dense, column-major implementation
 */
public interface TokenMatrix {
    /**
     * Creates a new dense matrix of the provided dimensions, with every value initialized to
     * {@code 0}.
     * @param rows The number of rows
     * @param columns The number of columns
     * @return The new matrix
     * @throws IllegalArgumentException When either dimension is negative, or their product overflows
     */
    static TokenMatrix of(int rows, int columns) {
        return new DenseTokenMatrix(rows, columns);
    }

    /**
     * Creates a new dense matrix of the provided dimensions, with every value initialized to
     * {@code 0}.
     * @param size The dimensions of the new matrix
     * @return The new matrix
     * @throws IllegalArgumentException When either dimension is negative, or their product overflows
     * @throws NullPointerException When the provided size is {@code null}
     */
    static TokenMatrix of(Dimension size) {
        return new DenseTokenMatrix(size);
    }

    /**
     * Creates a deep copy of the provided matrix, where any later change to the source is not
     * reflected in the copy.
     * @param m The source to copy from
     * @return The new matrix
     * @throws NullPointerException When the source is {@code null}
     */
    static TokenMatrix copyOf(TokenMatrix m) {
        return new DenseTokenMatrix(m);
    }

    /**
     * Returns the number of rows of this matrix.
     * @return The number of rows
     */
    int rows();

    /**
     * Returns the number of columns of this matrix.
     * @return The number of columns
     */
    int columns();

    /**
     * Returns the value at the provided row and column.
     * @param r The row to get
     * @param c The column to get
     * @return The value
     * @throws IndexOutOfBoundsException When either coordinate is out of bounds
     */
    float get(int r, int c) throws IndexOutOfBoundsException;

    /**
     * Sets the value at the provided row and column.
     * @param r The row to set
     * @param c The column to set
     * @param v The value to set to
     * @throws IndexOutOfBoundsException When either coordinate is out of bounds
     */
    void set(int r, int c, float v) throws IndexOutOfBoundsException;

    /**
     * Returns the dimensions of this matrix.
     * @return The dimensions of this matrix
     */
    default Dimension size() {
        return Dimension.rowCols(rows(), columns());
    }

    /**
     * Returns the total number of values held by this matrix.
     * @return The number of values
     */
    default int count() {
        return rows() * columns();
    }

    /**
     * Returns the contents of this matrix as a two-dimensional array, indexed by row then column.
     * @return The contents of this matrix
     */
    default float[][] toFloatArray() {
        int rows = rows();
        int columns = columns();
        float[][] result = new float[rows][columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[r][c] = get(r, c);
            }
        }

        return result;
    }

    /**
     * Returns the contents of this matrix as a two-dimensional array of strings, indexed by row then
     * column, using {@link Object#toString()} to format each value.
     * @return The contents of this matrix
     */
    default String[][] toStringArray() {
        return toStringArray(Object::toString);
    }

    /**
     * Returns the contents of this matrix as a two-dimensional array of strings, indexed by row then
     * column.
     * @param formatter The formatter to convert each value with
     * @return The contents of this matrix
     * @throws NullPointerException When the formatter is {@code null}
     */
    default String[][] toStringArray(Function<Float, String> formatter) {
        Objects.requireNonNull(formatter, "Formatter cannot be null.");

        int rows = rows();
        int columns = columns();
        String[][] result = new String[rows][columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[r][c] = formatter.apply(get(r, c));
            }
        }

        return result;
    }

    /**
     * Compares two matrices for equality by shape and contents, regardless of their
     * implementations. Every implementation must be consistent with this, so that a dense matrix
     * and any other implementation holding the same values compare equal.
     * @param a The first matrix, which may be {@code null}
     * @param b The second matrix, which may be {@code null}
     * @return {@code true} if both are {@code null}, or both have equal dimensions and contents
     */
    static boolean equals(TokenMatrix a, TokenMatrix b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a.rows() != b.rows() || a.columns() != b.columns()) return false;

        for (int c = 0; c < a.columns(); c++) {
            for (int r = 0; r < a.rows(); r++) {
                if (Float.compare(a.get(r, c), b.get(r, c)) != 0) return false;
            }
        }

        return true;
    }

    /**
     * Computes the hash code of a matrix from its shape and contents. Every implementation must
     * use this, so that two equal matrices of differing implementations hash alike.
     * @param m The matrix to hash
     * @return The hash code of the provided matrix
     * @throws NullPointerException When the matrix is {@code null}
     */
    static int hashCode(TokenMatrix m) {
        Objects.requireNonNull(m, "Cannot hash a null matrix.");

        int result = 1;
        result = 31 * result + m.rows();
        result = 31 * result + m.columns();

        for (int c = 0; c < m.columns(); c++) {
            for (int r = 0; r < m.rows(); r++) {
                result = 31 * result + Float.hashCode(m.get(r, c));
            }
        }

        return result;
    }
}
