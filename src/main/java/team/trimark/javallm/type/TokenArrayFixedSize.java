package team.trimark.javallm.type;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

/**
 * An array of tokens whose length is fixed at construction time. Unlike {@link TokenArray}, this
 * implementation never grows or shrinks, and every index within {@link #size()} is always readable.
 */
public class TokenArrayFixedSize implements TokenArrayCompatible {
    /**
     * The backing values of this array.
     */
    public final float[] values;

    /**
     * Creates a new token array of the provided size, with every value initialized to {@code 0}.
     * @param size The size of the new array
     * @throws IllegalArgumentException When the size is negative
     */
    public TokenArrayFixedSize(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("TokenArrayFixedSize size cannot be negative.");
        }

        this.values = new float[size];
    }

    /**
     * Creates a new token array holding a copy of the provided values. Later changes to the source
     * array are not reflected in the new instance.
     * @param values The values to copy
     * @return The new token array
     * @throws NullPointerException When the provided array is {@code null}
     */
    public static TokenArrayFixedSize copyOf(float... values) {
        Objects.requireNonNull(values, "Array cannot be null.");
        return new TokenArrayFixedSize(Arrays.copyOf(values, values.length));
    }

    /**
     * Creates a new token array backed directly by the provided array. Changes made through either
     * reference are visible to the other.
     * @param values The array to wrap
     * @return The new token array
     * @throws NullPointerException When the provided array is {@code null}
     */
    public static TokenArrayFixedSize refOf(float[] values) {
        Objects.requireNonNull(values, "Array cannot be null.");
        return new TokenArrayFixedSize(values);
    }

    /**
     * Creates a new token array directly backed by the provided array.
     * @param values The array to wrap
     */
    private TokenArrayFixedSize(float[] values) {
        this.values = Objects.requireNonNull(values);
    }

    /**
     * Creates a new token array from another one. Produces a deep copy, where any change from the
     * source will not be reflected in the new instance.
     * @param tac The source to copy from
     * @throws NullPointerException When the source is {@code null}
     */
    public TokenArrayFixedSize(TokenArrayCompatible tac) {
        this(Objects.requireNonNull(tac, "Source TokenArray cannot be null.").size());

        if (tac instanceof TokenArrayFixedSize tafs) {
            System.arraycopy(tafs.values, 0, this.values, 0, values.length);
        } else {
            System.arraycopy(tac.toArray(), 0, this.values, 0, values.length);
        }
    }

    /**
     * Returns a real-time iterator, not thread-safe.
     * @return The iterator
     */
    @Override
    public Iterator<Float> iterator() {
        return new TokenArrayLiveIterator(values.length, values);
    }

    /**
     * Returns the size of this token array, which never changes.
     * @return The size of this token array
     */
    @Override
    public int size() {
        return values.length;
    }

    /**
     * Returns the {@code i}th value of this array.
     * @param i The index to get
     * @return The value
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    @Override
    public float get(int i) throws IndexOutOfBoundsException {
        return values[i];
    }

    /**
     * Sets the {@code i}th value of this array.
     * @param i The index to set
     * @param value The value to set to
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    @Override
    public void set(int i, float value) throws IndexOutOfBoundsException {
        values[i] = value;
    }

    /**
     * Bulk sets this array, writing the provided values starting at index {@code i}. Since this array
     * cannot grow, the payload must fit entirely within the existing bounds.
     * @param i The index to start setting at
     * @param values The values to overwrite with
     * @throws IndexOutOfBoundsException When the index is out of bounds, or the payload does not fit
     * @throws NullPointerException When the provided array is {@code null}
     */
    @Override
    public void set(int i, float[] values) throws IndexOutOfBoundsException {
        int payloadSize = Objects.requireNonNull(values, "Cannot be null.").length;

        if (i < 0 || i > this.values.length) {
            throw new IndexOutOfBoundsException("Invalid index " + i + " for size of " + this.values.length);
        }

        if (payloadSize > this.values.length - i) {
            throw new IndexOutOfBoundsException("Array is too large to fit.");
        }

        System.arraycopy(values, 0, this.values, i, payloadSize);
    }

    /**
     * Resets every value of this array to {@code 0}. The size is left unchanged.
     */
    @Override
    public void clear() {
        Arrays.fill(values, 0f);
    }

    /**
     * Returns the contents of this array.
     * @return The contents of this array
     */
    @Override
    public float[] toArray() {
        return Arrays.copyOf(values, values.length);
    }
}
