package team.trimark.javallm.nn;

import team.trimark.javallm.type.TokenMatrix;

import java.util.Random;

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
     * Creates a matrix of the provided shape filled with samples from a normal distribution.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param stdDev The standard deviation of the distribution
     * @param random The source of randomness
     * @return The new matrix
     */
    public static TokenMatrix randomNormal(int rows, int columns, double stdDev, Random random) {
        TokenMatrix out = new TokenMatrix(rows, columns);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                out.set(r, c, (float) (random.nextGaussian() * stdDev));
            }
        }

        return out;
    }

    /**
     * Creates a matrix of the provided shape with every entry set to the provided value.
     * @param rows The number of rows
     * @param columns The number of columns
     * @param value The value to fill with
     * @return The new matrix
     */
    public static TokenMatrix filled(int rows, int columns, float value) {
        TokenMatrix out = new TokenMatrix(rows, columns);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                out.set(r, c, value);
            }
        }

        return out;
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

        TokenMatrix out = new TokenMatrix(n, p);

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < m; k++) {
                float av = a.get(i, k);

                for (int j = 0; j < p; j++) {
                    out.set(i, j, out.get(i, j) + av * b.get(k, j));
                }
            }
        }

        return out;
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

        TokenMatrix out = new TokenMatrix(n, p);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0f;

                for (int k = 0; k < m; k++) {
                    sum += a.get(i, k) * b.get(j, k);
                }

                out.set(i, j, sum);
            }
        }

        return out;
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

        TokenMatrix out = new TokenMatrix(m, p);

        for (int k = 0; k < n; k++) {
            for (int i = 0; i < m; i++) {
                float av = a.get(k, i);

                for (int j = 0; j < p; j++) {
                    out.set(i, j, out.get(i, j) + av * b.get(k, j));
                }
            }
        }

        return out;
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
        TokenMatrix out = new TokenMatrix(a.rows(), a.columns());

        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                out.set(r, c, a.get(r, c) + b.get(r, c));
            }
        }

        return out;
    }

    /**
     * Adds every entry of {@code source} into the matching entry of {@code target}, in place.
     * @param target The matrix to accumulate into
     * @param source The matrix to accumulate from
     * @throws IllegalArgumentException When the shapes disagree
     */
    public static void addInPlace(TokenMatrix target, TokenMatrix source) {
        requireSameShape(target, source);

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

        TokenMatrix out = new TokenMatrix(a.rows(), a.columns());

        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                out.set(r, c, a.get(r, c) + rowVector.get(0, c));
            }
        }

        return out;
    }

    /**
     * Sums every row of a matrix into a single row vector.
     * @param a The matrix, of shape {@code n * m}
     * @return The column-wise sums, of shape {@code 1 * m}
     */
    public static TokenMatrix sumRows(TokenMatrix a) {
        TokenMatrix out = new TokenMatrix(1, a.columns());

        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                out.set(0, c, out.get(0, c) + a.get(r, c));
            }
        }

        return out;
    }

    /**
     * Multiplies every entry of a matrix by a scalar, returning a new matrix.
     * @param a The matrix
     * @param scalar The scalar to multiply by
     * @return The scaled matrix
     */
    public static TokenMatrix scale(TokenMatrix a, float scalar) {
        TokenMatrix out = new TokenMatrix(a.rows(), a.columns());

        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                out.set(r, c, a.get(r, c) * scalar);
            }
        }

        return out;
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
        TokenMatrix out = new TokenMatrix(a.rows(), a.columns());

        for (int r = 0; r < a.rows(); r++) {
            for (int c = 0; c < a.columns(); c++) {
                out.set(r, c, a.get(r, c) * b.get(r, c));
            }
        }

        return out;
    }

    /**
     * Applies a numerically stable softmax independently to each row of a matrix.
     * @param a The matrix of scores
     * @return The matrix of probabilities, where every row sums to {@code 1}
     */
    public static TokenMatrix softmaxRows(TokenMatrix a) {
        TokenMatrix out = new TokenMatrix(a.rows(), a.columns());

        for (int r = 0; r < a.rows(); r++) {
            float max = Float.NEGATIVE_INFINITY;

            for (int c = 0; c < a.columns(); c++) {
                max = Math.max(max, a.get(r, c));
            }

            float sum = 0f;

            for (int c = 0; c < a.columns(); c++) {
                float e = (float) Math.exp(a.get(r, c) - max);
                out.set(r, c, e);
                sum += e;
            }

            for (int c = 0; c < a.columns(); c++) {
                out.set(r, c, out.get(r, c) / sum);
            }
        }

        return out;
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
