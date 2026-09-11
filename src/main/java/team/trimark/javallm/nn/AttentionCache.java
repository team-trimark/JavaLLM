package team.trimark.javallm.nn;

import team.trimark.javallm.type.TokenMatrix;

/**
 * The keys and values already computed by one {@link CausalSelfAttention} head, retained
 * so that generating a token does not recompute them.
 * <p>
 * A causal model never revises the past: the key and value a position produces depend only
 * on that position and the ones before it, so once computed they are fixed for the rest of
 * the sequence. Without this store, emitting each new token re-runs attention over the
 * whole window, which costs {@code O(t^2)} per token instead of {@code O(t)}.
 * <p>
 * This holds inference state only. It is never consulted during training and holds nothing
 * needed by a backward pass.
 */
public final class AttentionCache {
    /**
     * The key of every position retained so far, one per row.
     */
    private final TokenMatrix keys;

    /**
     * The value of every position retained so far, one per row.
     */
    private final TokenMatrix values;

    /**
     * The number of positions currently held.
     */
    private int length;

    /**
     * Creates an empty cache with room for the provided number of positions.
     * @param capacity The greatest number of positions to retain
     * @param features The number of features per position
     * @throws IllegalArgumentException When either argument is not positive
     */
    public AttentionCache(int capacity, int features) {
        if (capacity <= 0 || features <= 0) {
            throw new IllegalArgumentException("Cache capacity and features must be positive.");
        }

        this.keys = TokenMatrix.of(capacity, features);
        this.values = TokenMatrix.of(capacity, features);
        this.length = 0;
    }

    /**
     * Returns the number of positions currently held.
     * @return The number of positions
     */
    public int length() {
        return length;
    }

    /**
     * Returns the greatest number of positions this cache can hold.
     * @return The capacity
     */
    public int capacity() {
        return keys.rows();
    }

    /**
     * Discards every retained position, so the next append starts a fresh sequence.
     */
    public void clear() {
        length = 0;
    }

    /**
     * Appends the key and value of one position.
     * @param key The key, of shape {@code 1 * features}
     * @param value The value, of shape {@code 1 * features}
     * @throws IllegalStateException When the cache is already full
     */
    public void append(TokenMatrix key, TokenMatrix value) {
        if (length >= keys.rows()) {
            throw new IllegalStateException("Attention cache is full at " + length + " positions.");
        }

        for (int c = 0; c < keys.columns(); c++) {
            keys.set(length, c, key.get(0, c));
            values.set(length, c, value.get(0, c));
        }

        length++;
    }

    /**
     * Returns the {@code c}th feature of the key at the provided position.
     * @param position The position to read
     * @param c The feature to read
     * @return The value
     */
    float key(int position, int c) {
        return keys.get(position, c);
    }

    /**
     * Returns the {@code c}th feature of the value at the provided position.
     * @param position The position to read
     * @param c The feature to read
     * @return The value
     */
    float value(int position, int c) {
        return values.get(position, c);
    }
}
