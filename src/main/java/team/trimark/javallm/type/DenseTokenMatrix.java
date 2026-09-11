package team.trimark.javallm.type;

import java.util.Arrays;
import java.util.Objects;

/**
 * A {@link TokenMatrix} holding every value explicitly in one flat array, in column-major order:
 * the value at {@code (r, c)} lives at index {@code c * rows + r}.
 * <p>
 * Column-major is not an arbitrary choice. The product kernels accumulate one output column at a
 * time, walking down the rows of that column in their innermost loop; storing a column
 * contiguously is what makes that loop a linear scan of memory rather than a stride of
 * {@code columns} floats per step. The layout is chosen to match the access pattern of the code
 * that runs hottest.
 * <p>
 * Because the layout is part of this class's published behaviour rather than an implementation
 * detail, bulk callers can take the backing array through {@link #columnMajorValues()} and work
 * on it directly, and hand a finished array back through {@link #wrap(int, int, float[])}. Both
 * share rather than copy, which is the point: a kernel that copies its operands in and its result
 * out does three full passes over memory that produce no arithmetic.
 */
public final class DenseTokenMatrix implements TokenMatrix {
    /**
     * The number of rows of this matrix.
     */
    private final int rows;

    /**
     * The number of columns of this matrix.
     */
    private final int columns;

    /**
     * The flat backing values of this matrix, in column-major order.
     */
    private final float[] values;

    /**
     * Creates a new matrix of the provided dimensions, with every value initialized to {@code 0}.
     * @param rows The number of rows
     * @param columns The number of columns
     * @throws IllegalArgumentException When either dimension is negative, or their product overflows
     */
    public DenseTokenMatrix(int rows, int columns) {
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
    public DenseTokenMatrix(Dimension size) {
        Objects.requireNonNull(size, "Size of a matrix cannot be null.");

        int r = size.rows();
        int c = size.columns();

        if (r < 0 || c < 0 || (long) r * c > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Dimensions are invalid or too large: " + size);
        }

        this.rows = r;
        this.columns = c;
        this.values = new float[r * c];
    }

    /**
     * Creates a new matrix from another one. Produces a deep copy, where any change from the source
     * will not be reflected in the new instance.
     * @param m The source to copy from
     * @throws NullPointerException When the source is {@code null}
     */
    public DenseTokenMatrix(TokenMatrix m) {
        Objects.requireNonNull(m, "Cannot copy from null matrix.");

        this.rows = m.rows();
        this.columns = m.columns();
        this.values = new float[rows * columns];

        if (m instanceof DenseTokenMatrix dense) {
            System.arraycopy(dense.values, 0, values, 0, values.length);
            return;
        }

        for (int c = 0; c < columns; c++) {
            int offset = c * rows;

            for (int r = 0; r < rows; r++) {
                values[offset + r] = m.get(r, c);
            }
        }
    }

    /**
     * Creates a matrix that takes ownership of the provided array rather than copying it. The
     * caller must not retain or mutate the array afterwards; it becomes the backing store of the
     * returned matrix.
     * <p>
     * This exists for kernels that have just computed a result into a scratch array of the right
     * layout. Wrapping it costs nothing, where copying it costs a full pass over memory.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param values The values to adopt, in column-major order, of length {@code rows * columns}
     * @return The new matrix, backed by the provided array
     * @throws NullPointerException When the array is {@code null}
     * @throws IllegalArgumentException When either dimension is negative, or the array length does
     *         not match the shape
     */
    public static DenseTokenMatrix wrap(int rows, int columns, float[] values) {
        Objects.requireNonNull(values, "Cannot wrap a null array.");

        if (rows < 0 || columns < 0) {
            throw new IllegalArgumentException("Dimensions are invalid: " + Dimension.rowCols(rows, columns));
        }

        if ((long) rows * columns != values.length) {
            throw new IllegalArgumentException(
                    "Array of length " + values.length + " does not fill " + rows + " * " + columns + ".");
        }

        return new DenseTokenMatrix(rows, columns, values);
    }

    /**
     * Creates a matrix backed directly by the provided array, whose arguments have already been
     * checked.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param values The values to adopt, in column-major order
     */
    private DenseTokenMatrix(int rows, int columns, float[] values) {
        this.rows = rows;
        this.columns = columns;
        this.values = values;
    }

    /**
     * Returns the backing array of this matrix, in column-major order, where the value at
     * {@code (r, c)} lives at index {@code c * rows + r}.
     * <p>
     * This is the live array, not a copy. Writing to it writes through to this matrix, which is
     * intended for bulk kernels and a hazard for everybody else.
     * @return The backing values of this matrix
     */
    public float[] columnMajorValues() {
        return values;
    }

    /**
     * Returns the number of rows of this matrix.
     * @return The number of rows
     */
    @Override
    public int rows() {
        return rows;
    }

    /**
     * Returns the number of columns of this matrix.
     * @return The number of columns
     */
    @Override
    public int columns() {
        return columns;
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
    @Override
    public float get(int r, int c) throws IndexOutOfBoundsException {
        if (r >= rows || c >= columns || r < 0 || c < 0) {
            throw new IndexOutOfBoundsException("Index out of bounds for matrix " + rows + " * " + columns);
        }

        return values[rowColToFlat(r, c)];
    }

    /**
     * Sets the value at the provided row and column.
     * @param r The row to set
     * @param c The column to set
     * @param v The value to set to
     * @throws IndexOutOfBoundsException When either coordinate is out of bounds
     */
    @Override
    public void set(int r, int c, float v) throws IndexOutOfBoundsException {
        if (r >= rows || c >= columns || r < 0 || c < 0) {
            throw new IndexOutOfBoundsException("Index out of bounds for matrix " + rows + " * " + columns);
        }

        values[rowColToFlat(r, c)] = v;
    }

    /**
     * Sets every value of this matrix to the provided value.
     * @param v The value to fill with
     */
    public void fill(float v) {
        Arrays.fill(values, v);
    }

    /**
     * Returns the total number of values held by this matrix.
     * @return The number of values
     */
    @Override
    public int count() {
        return values.length;
    }

    /**
     * Checks for equality with another object.
     * @param obj The object to compare to
     * @return {@code true} if the other object is a matrix of equal dimensions and contents
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj instanceof DenseTokenMatrix dense) {
            return rows == dense.rows && columns == dense.columns && Arrays.equals(values, dense.values);
        }

        return obj instanceof TokenMatrix other && TokenMatrix.equals(this, other);
    }

    /**
     * Returns the hash code of this matrix.
     * @return The hash code of this matrix
     */
    @Override
    public int hashCode() {
        return TokenMatrix.hashCode(this);
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
