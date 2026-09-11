package team.trimark.javallm.nn;

import team.trimark.javallm.type.DenseTokenMatrix;
import team.trimark.javallm.type.TokenMatrix;

import java.util.Objects;

/**
 * A trainable tensor paired with the gradient accumulated for it during backpropagation.
 */
public final class Parameter {
    /**
     * The human readable name of this parameter, used for diagnostics.
     */
    public final String name;

    /**
     * The current values of this parameter.
     */
    public final TokenMatrix value;

    /**
     * The gradient of the loss with respect to this parameter.
     */
    public final TokenMatrix gradient;

    /**
     * Creates a new parameter wrapping the provided values.
     * @param name The name of the parameter
     * @param value The initial values
     * @throws NullPointerException When either argument is {@code null}
     */
    public Parameter(String name, TokenMatrix value) {
        this.name = Objects.requireNonNull(name, "Parameter name cannot be null.");
        this.value = Objects.requireNonNull(value, "Parameter value cannot be null.");
        this.gradient = TokenMatrix.of(value.rows(), value.columns());
    }

    /**
     * Resets the accumulated gradient of this parameter to zero.
     */
    public void zeroGradient() {
        if (gradient instanceof DenseTokenMatrix dense) {
            dense.fill(0f);
            return;
        }

        for (int r = 0; r < gradient.rows(); r++) {
            for (int c = 0; c < gradient.columns(); c++) {
                gradient.set(r, c, 0f);
            }
        }
    }

    /**
     * Returns the total number of scalar values held by this parameter.
     * @return The number of scalar values
     */
    public int count() {
        return value.rows() * value.columns();
    }
}
