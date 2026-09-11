package team.trimark.javallm.nn;

import team.trimark.javallm.type.TokenMatrix;

import java.util.List;

/**
 * Single-head causal self-attention. Every position builds a query, a key and a value from its own
 * activations, scores its query against the keys of all positions at or before it, and reads out a
 * weighted average of the corresponding values. The causal mask is what makes the model a language
 * model: no position may ever look at a token that comes after it.
 */
public final class CausalSelfAttention {
    /**
     * The query projection.
     */
    private final Parameter queryWeights;

    /**
     * The key projection.
     */
    private final Parameter keyWeights;

    /**
     * The value projection.
     */
    private final Parameter valueWeights;

    /**
     * The output projection applied after the values have been mixed.
     */
    private final Parameter outputWeights;

    /**
     * The scaling applied to the raw scores before the softmax.
     */
    private final float scale;

    /**
     * The input of the most recent forward pass.
     */
    private TokenMatrix input;

    /**
     * The queries of the most recent forward pass.
     */
    private TokenMatrix queries;

    /**
     * The keys of the most recent forward pass.
     */
    private TokenMatrix keys;

    /**
     * The values of the most recent forward pass.
     */
    private TokenMatrix values;

    /**
     * The attention probabilities of the most recent forward pass.
     */
    private TokenMatrix attention;

    /**
     * The mixed values of the most recent forward pass, before the output projection.
     */
    private TokenMatrix mixed;

    /**
     * Creates a new attention head.
     * @param name The name prefix used for the parameters
     * @param features The number of features per position
     * @param parameters The registry to add the new parameters to
     * @param random The source of randomness for initialization
     */
    public CausalSelfAttention(String name, int features, List<Parameter> parameters, java.util.Random random) {
        double deviation = 0.02;

        this.queryWeights = new Parameter(name + ".wq", Matrices.randomNormal(features, features, deviation, random));
        this.keyWeights = new Parameter(name + ".wk", Matrices.randomNormal(features, features, deviation, random));
        this.valueWeights = new Parameter(name + ".wv", Matrices.randomNormal(features, features, deviation, random));
        this.outputWeights = new Parameter(name + ".wo", Matrices.randomNormal(features, features, deviation, random));

        this.scale = 1f / (float) Math.sqrt(features);

        parameters.add(queryWeights);
        parameters.add(keyWeights);
        parameters.add(valueWeights);
        parameters.add(outputWeights);
    }

    /**
     * Runs attention for a single new position, reading the keys and values of every
     * earlier position from the provided cache and appending its own.
     * <p>
     * This is the inference counterpart of {@link #forward(TokenMatrix)} and computes the
     * same result for the final row, but in {@code O(t)} rather than {@code O(t^2)}. It
     * records nothing for a backward pass, so it must not be used while training.
     * @param x The activations of the new position, of shape {@code 1 * features}
     * @param cache The keys and values of the positions before it
     * @return The attention output, of shape {@code 1 * features}
     * @throws IllegalStateException When the cache is full
     */
    public TokenMatrix forwardStep(TokenMatrix x, AttentionCache cache) {
        TokenMatrix query = Matrices.matmul(x, queryWeights.value);
        TokenMatrix key = Matrices.matmul(x, keyWeights.value);
        TokenMatrix value = Matrices.matmul(x, valueWeights.value);

        cache.append(key, value);

        int t = cache.length();
        int d = x.columns();
        float[] weights = new float[t];
        float max = Float.NEGATIVE_INFINITY;

        for (int j = 0; j < t; j++) {
            float dot = 0f;

            for (int c = 0; c < d; c++) {
                dot += query.get(0, c) * cache.key(j, c);
            }

            weights[j] = dot * scale;
            max = Math.max(max, weights[j]);
        }

        float sum = 0f;

        for (int j = 0; j < t; j++) {
            weights[j] = (float) Math.exp(weights[j] - max);
            sum += weights[j];
        }

        TokenMatrix mixedRow = TokenMatrix.of(1, d);

        for (int j = 0; j < t; j++) {
            float weight = weights[j] / sum;

            for (int c = 0; c < d; c++) {
                mixedRow.set(0, c, mixedRow.get(0, c) + weight * cache.value(j, c));
            }
        }

        return Matrices.matmul(mixedRow, outputWeights.value);
    }

    /**
     * Runs attention over the provided activations.
     * @param x The activations, of shape {@code sequence * features}
     * @return The attention output, of the same shape
     */
    public TokenMatrix forward(TokenMatrix x) {
        input = x;

        queries = Matrices.matmul(x, queryWeights.value);
        keys = Matrices.matmul(x, keyWeights.value);
        values = Matrices.matmul(x, valueWeights.value);

        int t = x.rows();
        TokenMatrix scores = Matrices.matmulNT(queries, keys);

        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                scores.set(i, j, j > i ? Float.NEGATIVE_INFINITY : scores.get(i, j) * scale);
            }
        }

        attention = Matrices.softmaxRows(scores);
        mixed = Matrices.matmul(attention, values);

        return Matrices.matmul(mixed, outputWeights.value);
    }

    /**
     * Propagates the gradient back through the most recent forward pass, accumulating the gradients
     * of the four projections.
     * @param dOut The gradient with respect to the output of this layer
     * @return The gradient with respect to the input of this layer
     */
    public TokenMatrix backward(TokenMatrix dOut) {
        int t = dOut.rows();

        Matrices.addInPlace(outputWeights.gradient, Matrices.matmulTN(mixed, dOut));
        TokenMatrix dMixed = Matrices.matmulNT(dOut, outputWeights.value);

        TokenMatrix dAttention = Matrices.matmulNT(dMixed, values);
        TokenMatrix dValues = Matrices.matmulTN(attention, dMixed);

        TokenMatrix dScores = TokenMatrix.of(t, t);

        for (int i = 0; i < t; i++) {
            float rowDot = 0f;

            for (int j = 0; j <= i; j++) {
                rowDot += dAttention.get(i, j) * attention.get(i, j);
            }

            for (int j = 0; j <= i; j++) {
                float a = attention.get(i, j);
                dScores.set(i, j, a * (dAttention.get(i, j) - rowDot) * scale);
            }
        }

        TokenMatrix dQueries = Matrices.matmul(dScores, keys);
        TokenMatrix dKeys = Matrices.matmulTN(dScores, queries);

        Matrices.addInPlace(queryWeights.gradient, Matrices.matmulTN(input, dQueries));
        Matrices.addInPlace(keyWeights.gradient, Matrices.matmulTN(input, dKeys));
        Matrices.addInPlace(valueWeights.gradient, Matrices.matmulTN(input, dValues));

        TokenMatrix dx = Matrices.matmulNT(dQueries, queryWeights.value);
        Matrices.addInPlace(dx, Matrices.matmulNT(dKeys, keyWeights.value));
        Matrices.addInPlace(dx, Matrices.matmulNT(dValues, valueWeights.value));

        return dx;
    }
}
