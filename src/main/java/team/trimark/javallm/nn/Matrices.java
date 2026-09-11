package team.trimark.javallm.nn;

import team.trimark.javallm.type.DenseTokenMatrix;
import team.trimark.javallm.type.TokenMatrix;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * Dense linear algebra operations over {@link TokenMatrix}. Every operation treats a matrix as
 * row-major in the mathematical sense: index {@code (r, c)} is row {@code r}, column {@code c}.
 */
public final class Matrices {
    /**
     * Private constructor to prevent instantiation.
     */
    private Matrices() {}

    /**
     * The number of output columns below which a product is computed on the calling thread.
     * Below this width the fork/join overhead costs more than the parallelism returns.
     */
    private static final int PARALLEL_THRESHOLD = 8;

    /**
     * Returns the values of a matrix as a flat array in column-major order, so that the product
     * kernels can walk memory contiguously as the {@code get}/{@code set} accessors cannot.
     * <p>
     * A {@link DenseTokenMatrix} already stores its values in exactly this layout, so its backing
     * array is returned as-is and no copy is made. Only a foreign implementation has to be copied.
     * The returned array must be treated as read-only: when it is shared, writing to it would
     * write through to the operand.
     * @param matrix The matrix to read
     * @return The values, indexed by {@code column * rows + row}, possibly shared with the matrix
     */
    private static float[] flat(TokenMatrix matrix) {
        if (matrix instanceof DenseTokenMatrix dense) {
            return dense.columnMajorValues();
        }

        int rows = matrix.rows();
        int columns = matrix.columns();
        float[] values = new float[rows * columns];

        for (int c = 0; c < columns; c++) {
            int offset = c * rows;

            for (int r = 0; r < rows; r++) {
                values[offset + r] = matrix.get(r, c);
            }
        }

        return values;
    }

    /**
     * Builds a matrix of the provided shape backed by a freshly computed column-major array. The
     * array is adopted rather than copied, so the caller must not retain it.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param values The values, indexed by {@code column * rows + row}
     * @return The new matrix
     */
    private static TokenMatrix unflatten(int rows, int columns, float[] values) {
        return DenseTokenMatrix.wrap(rows, columns, values);
    }

    /**
     * Runs the provided action once for every output column, in parallel when there are
     * enough of them to be worth it. Every column of a matrix product is independent, so
     * no two invocations ever write to the same element and no synchronisation is needed.
     * @param columns The number of columns to cover
     * @param action The action to run for each column index
     */
    private static void forEachColumn(int columns, java.util.function.IntConsumer action) {
        if (columns < PARALLEL_THRESHOLD) {
            for (int j = 0; j < columns; j++) {
                action.accept(j);
            }

            return;
        }

        IntStream.range(0, columns).parallel().forEach(action);
    }

    /**
     * Creates a matrix of the provided shape filled with samples from a normal distribution.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param stdDev The standard deviation of the distribution
     * @param random The source of randomness
     * @return The new matrix
     */
    public static TokenMatrix randomNormal(int rows, int columns, double stdDev, Random random) {
        float[] out = new float[rows * columns];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                out[c * rows + r] = (float) (random.nextGaussian() * stdDev);
            }
        }

        return DenseTokenMatrix.wrap(rows, columns, out);
    }

    /**
     * Creates a matrix of the provided shape with every entry set to the provided value.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param value The value to fill with
     * @return The new matrix
     */
    public static TokenMatrix filled(int rows, int columns, float value) {
        float[] out = new float[rows * columns];
        java.util.Arrays.fill(out, value);
        return DenseTokenMatrix.wrap(rows, columns, out);
    }

    /**
     * Computes the matrix product {@code a * b}.
     * @param a The left operand, of shape {@code n * m}
     * @param b The right operand, of shape {@code m * p}
     * @return The product, of shape {@code n * p}
     * @throws IllegalArgumentException When the inner dimensions disagree
     */
    public static TokenMatrix matmul(TokenMatrix a, TokenMatrix b) {
        int n = a.rows();
        int m = a.columns();
        int p = b.columns();

        if (m != b.rows()) {
            throw new IllegalArgumentException("Cannot multiply " + n + "*" + m + " by " + b.rows() + "*" + p);
        }

        float[] av = flat(a);
        float[] bv = flat(b);
        float[] ov = new float[n * p];

        forEachColumn(p, j -> {
            int outOffset = j * n;
            int rightOffset = j * m;

            for (int k = 0; k < m; k++) {
                float scalar = bv[rightOffset + k];

                if (scalar == 0f) {
                    continue;
                }

                int leftOffset = k * n;

                for (int i = 0; i < n; i++) {
                    ov[outOffset + i] += av[leftOffset + i] * scalar;
                }
            }
        });

        return unflatten(n, p, ov);
    }

    /**
     * Computes the matrix product {@code a * transpose(b)}.
     * @param a The left operand, of shape {@code n * m}
     * @param b The right operand, of shape {@code p * m}
     * @return The product, of shape {@code n * p}
     * @throws IllegalArgumentException When the inner dimensions disagree
     */
    public static TokenMatrix matmulNT(TokenMatrix a, TokenMatrix b) {
        int n = a.rows();
        int m = a.columns();
        int p = b.rows();

        if (m != b.columns()) {
            throw new IllegalArgumentException("Cannot multiply " + n + "*" + m + " by transposed " + p + "*" + b.columns());
        }

        float[] av = flat(a);
        float[] bv = flat(b);
        float[] ov = new float[n * p];

        forEachColumn(p, j -> {
            int outOffset = j * n;

            for (int k = 0; k < m; k++) {
                float scalar = bv[k * p + j];

                if (scalar == 0f) {
                    continue;
                }

                int leftOffset = k * n;

                for (int i = 0; i < n; i++) {
                    ov[outOffset + i] += av[leftOffset + i] * scalar;
                }
            }
        });

        return unflatten(n, p, ov);
    }

    /**
     * Computes the matrix product {@code transpose(a) * b}.
     * @param a The left operand, of shape {@code n * m}
     * @param b The right operand, of shape {@code n * p}
     * @return The product, of shape {@code m * p}
     * @throws IllegalArgumentException When the outer dimensions disagree
     */
    public static TokenMatrix matmulTN(TokenMatrix a, TokenMatrix b) {
        int n = a.rows();
        int m = a.columns();
        int p = b.columns();

        if (n != b.rows()) {
            throw new IllegalArgumentException("Cannot multiply transposed " + n + "*" + m + " by " + b.rows() + "*" + p);
        }

        float[] av = flat(a);
        float[] bv = flat(b);
        float[] ov = new float[m * p];

        forEachColumn(p, j -> {
            int outOffset = j * m;
            int rightOffset = j * n;

            int limit = n - (n % 4);

            for (int i = 0; i < m; i++) {
                int leftOffset = i * n;

                // Four accumulators rather than one. A single running sum makes every addition wait
                // for the previous one to land, which pins this loop to the latency of floating point
                // addition no matter how much the processor could otherwise overlap. Four independent
                // chains fill that idle time. Floating point addition is not associative, so this does
                // change the last bits of the result - the sum is exact to the same precision, but it
                // is a different order of it.
                float sum0 = 0f;
                float sum1 = 0f;
                float sum2 = 0f;
                float sum3 = 0f;

                int k = 0;

                for (; k < limit; k += 4) {
                    sum0 += av[leftOffset + k] * bv[rightOffset + k];
                    sum1 += av[leftOffset + k + 1] * bv[rightOffset + k + 1];
                    sum2 += av[leftOffset + k + 2] * bv[rightOffset + k + 2];
                    sum3 += av[leftOffset + k + 3] * bv[rightOffset + k + 3];
                }

                float sum = (sum0 + sum1) + (sum2 + sum3);

                for (; k < n; k++) {
                    sum += av[leftOffset + k] * bv[rightOffset + k];
                }

                ov[outOffset + i] = sum;
            }
        });

        return unflatten(m, p, ov);
    }

    /**
     * Computes the element-wise sum of two matrices of equal shape.
     * @param a The left operand
     * @param b The right operand
     * @return The sum
     * @throws IllegalArgumentException When the shapes disagree
     */
    public static TokenMatrix add(TokenMatrix a, TokenMatrix b) {
        requireSameShape(a, b);

        float[] av = flat(a);
        float[] bv = flat(b);
        float[] ov = new float[av.length];

        for (int i = 0; i < ov.length; i++) {
            ov[i] = av[i] + bv[i];
        }

        return unflatten(a.rows(), a.columns(), ov);
    }

    /**
     * Adds every entry of {@code source} into the matching entry of {@code target}, in place.
     * @param target The matrix to accumulate into
     * @param source The matrix to accumulate from
     * @throws IllegalArgumentException When the shapes disagree
     */
    public static void addInPlace(TokenMatrix target, TokenMatrix source) {
        requireSameShape(target, source);

        if (target instanceof DenseTokenMatrix dense) {
            float[] tv = dense.columnMajorValues();
            float[] sv = flat(source);

            for (int i = 0; i < tv.length; i++) {
                tv[i] += sv[i];
            }

            return;
        }

        for (int r = 0; r < target.rows(); r++) {
            for (int c = 0; c < target.columns(); c++) {
                target.set(r, c, target.get(r, c) + source.get(r, c));
            }
        }
    }

    /**
     * Adds a single row vector to every row of a matrix, returning a new matrix.
     * @param a The matrix, of shape {@code n * m}
     * @param rowVector The row vector, of shape {@code 1 * m}
     * @return The result, of shape {@code n * m}
     * @throws IllegalArgumentException When the widths disagree, or the vector is not a single row
     */
    public static TokenMatrix addRowVector(TokenMatrix a, TokenMatrix rowVector) {
        if (rowVector.rows() != 1 || rowVector.columns() != a.columns()) {
            throw new IllegalArgumentException("Expected a 1*" + a.columns() + " row vector.");
        }

        int rows = a.rows();
        int columns = a.columns();

        float[] av = flat(a);
        float[] vv = flat(rowVector);
        float[] ov = new float[av.length];

        for (int c = 0; c < columns; c++) {
            int offset = c * rows;
            float add = vv[c];

            for (int r = 0; r < rows; r++) {
                ov[offset + r] = av[offset + r] + add;
            }
        }

        return unflatten(rows, columns, ov);
    }

    /**
     * Sums every row of a matrix into a single row vector.
     * @param a The matrix, of shape {@code n * m}
     * @return The column-wise sums, of shape {@code 1 * m}
     */
    public static TokenMatrix sumRows(TokenMatrix a) {
        int rows = a.rows();
        int columns = a.columns();

        float[] av = flat(a);
        float[] ov = new float[columns];

        for (int c = 0; c < columns; c++) {
            int offset = c * rows;
            float sum = 0f;

            for (int r = 0; r < rows; r++) {
                sum += av[offset + r];
            }

            ov[c] = sum;
        }

        return unflatten(1, columns, ov);
    }

    /**
     * Multiplies every entry of a matrix by a scalar, returning a new matrix.
     * @param a The matrix
     * @param scalar The scalar to multiply by
     * @return The scaled matrix
     */
    public static TokenMatrix scale(TokenMatrix a, float scalar) {
        float[] av = flat(a);
        float[] ov = new float[av.length];

        for (int i = 0; i < ov.length; i++) {
            ov[i] = av[i] * scalar;
        }

        return unflatten(a.rows(), a.columns(), ov);
    }

    /**
     * Computes the element-wise product of two matrices of equal shape.
     * @param a The left operand
     * @param b The right operand
     * @return The element-wise product
     * @throws IllegalArgumentException When the shapes disagree
     */
    public static TokenMatrix multiply(TokenMatrix a, TokenMatrix b) {
        requireSameShape(a, b);

        float[] av = flat(a);
        float[] bv = flat(b);
        float[] ov = new float[av.length];

        for (int i = 0; i < ov.length; i++) {
            ov[i] = av[i] * bv[i];
        }

        return unflatten(a.rows(), a.columns(), ov);
    }

    /**
     * Applies a numerically stable softmax independently to each row of a matrix.
     * @param a The matrix of scores
     * @return The matrix of probabilities, where every row sums to {@code 1}
     */
    public static TokenMatrix softmaxRows(TokenMatrix a) {
        int rows = a.rows();
        int columns = a.columns();

        float[] av = flat(a);
        float[] ov = new float[av.length];

        for (int r = 0; r < rows; r++) {
            float max = Float.NEGATIVE_INFINITY;

            for (int c = 0; c < columns; c++) {
                max = Math.max(max, av[c * rows + r]);
            }

            float sum = 0f;

            for (int c = 0; c < columns; c++) {
                float e = (float) Math.exp(av[c * rows + r] - max);
                ov[c * rows + r] = e;
                sum += e;
            }

            for (int c = 0; c < columns; c++) {
                ov[c * rows + r] /= sum;
            }
        }

        return unflatten(rows, columns, ov);
    }

    /**
     * Throws when the two provided matrices do not have the same shape.
     * @param a The left operand
     * @param b The right operand
     * @throws IllegalArgumentException When the shapes disagree
     */
    private static void requireSameShape(TokenMatrix a, TokenMatrix b) {
        if (a.rows() != b.rows() || a.columns() != b.columns()) {
            throw new IllegalArgumentException("Shape mismatch: " + a.rows() + "*" + a.columns()
                    + " against " + b.rows() + "*" + b.columns());
        }
    }
}
