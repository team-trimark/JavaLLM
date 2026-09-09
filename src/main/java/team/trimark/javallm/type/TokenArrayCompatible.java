package team.trimark.javallm.type;

/**
 * The common contract shared by every array-like container of tokens.
 */
public interface TokenArrayCompatible extends Iterable<Float> {
    /**
     * Returns the number of readable tokens in this array.
     * @return The size of this array
     */
    int size();

    /**
     * Returns the {@code i}th value of this array.
     * @param i The index to get
     * @return The value
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    float get(int i) throws IndexOutOfBoundsException;

    /**
     * Sets the {@code i}th value of this array.
     * @param i The index to set
     * @param value The value to set to
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    void set(int i, float value) throws IndexOutOfBoundsException;

    /**
     * Bulk sets this array, writing the provided values starting at index {@code i}.
     * @param i The index to start setting at
     * @param values The values to overwrite with
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    void set(int i, float[] values) throws IndexOutOfBoundsException;

    /**
     * Clears this array, resetting its contents.
     */
    void clear();

    /**
     * Returns the contents of this array.
     * @return The contents of this array
     */
    float[] toArray();
}
