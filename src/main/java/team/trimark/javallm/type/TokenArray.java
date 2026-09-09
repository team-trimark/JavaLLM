package team.trimark.javallm.type;

import team.trimark.javallm.exception.OutOfEmptyIndexException;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

/**
 * An array of tokens.
 */
public class TokenArray implements Iterable<Float> {
    /**
     * The initial size when no specific size is designated.
     */
    private static final int DEFAULT_SIZE = 4096;

    /**
     * Block size.
     */
    private static final int BLOCK_SIZE = 1024;

    /**
     * Half of a block size.
     */
    private static final int HALF_BLOCK_SIZE = BLOCK_SIZE / 2;

    /**
     * Quarter of a block size.
     */
    private static final int QUARTER_BLOCK_SIZE = HALF_BLOCK_SIZE / 2;

    /**
     * Final block limit.
     */
    private static final int FINAL_BLOCK_LIMIT = Integer.MAX_VALUE - BLOCK_SIZE;

    private int size;
    private float[] values;

    /**
     * Creates a new token array with the size {@link #DEFAULT_SIZE}, nominally empty.
     */
    public TokenArray() {
        this(0);
    }

    /**
     * Creates a new token array with the provided initial size, nominally empty.
     * @param initialSize The initial size
     */
    public TokenArray(int initialSize) {
        if (initialSize < 0) {
            throw new IllegalArgumentException("TokenArray size cannot be negative.");
        }

        this.size = 0;
        this.values = new float[DEFAULT_SIZE];
    }

    /**
     * Creates a new token array from an array of floats.
     * @param values The values
     */
    public TokenArray(float[] values) {
        this.size = values.length;
        this.values = new float[size];

        System.arraycopy(values, 0, this.values, 0, size);
    }

    /**
     * Creates a new token array from another one. Produces a deep copy, where any change from the source will
     * not be reflected in the new instance.
     * @param ta The source to copy from
     */
    public TokenArray(TokenArray ta) {
        this(Objects.requireNonNull(ta, "Source TokenArray cannot be null.").size);
        System.arraycopy(ta.values, 0, this.values, 0, size);
    }

    /**
     * Returns the current capacity of this token array. A size of {@code 10} does not mean that there are
     * {@code 10} meaningful values, simply that there can be.
     * @return The current capacity of this token array
     */
    public int size() {
        return size;
    }

    /**
     * Returns the actual capacity this token array is using in memory.
     * @return The actual capacity of this token array
     */
    public int sizeActual() {
        return values.length;
    }

    /**
     * Resizes this token array nominally.
     * @param newSize The size to nominally resize to
     */
    public void resizeNominal(int newSize) {
        if (newSize < 0) {
            throw new IllegalArgumentException("TokenArray cannot be resized to a negative size.");
        }

        if (newSize == size) return; // Nothing to do

        int oldSizeActual = values.length;
        int oldSize = this.size;
        this.size = newSize;

        if (newSize < oldSizeActual) return; // Nothing left to do

        int bufferedNewSize = getBufferedNewSize(newSize);
        float[] newValues = new float[bufferedNewSize];
        System.arraycopy(values, 0, newValues, 0, oldSize);

        this.values = newValues;
    }

    /**
     * Resizes the actual size as well as the nominal size.
     * @param newSize The size to enforce
     */
    public void resizeActual(int newSize) {
        if (newSize < 0) {
            throw new IllegalArgumentException("TokenArray cannot be resized to a negative size.");
        }

        if (newSize == size && newSize == values.length) return; // Nothing to do

        if (values.length == newSize) {
            this.size = newSize;
            return; // Rewrite nominal size and exit
        }

        int oldSizeNominal = size;

        if (newSize > size) {
            // Increasing capacity
            size = newSize;
            float[] newValues = new float[newSize];
            System.arraycopy(values, 0, newValues, 0, oldSizeNominal);
            this.values = newValues;
        } else {
            // Decreasing capacity
            size = newSize;
            float[] newValues = new float[newSize];
            System.arraycopy(values, 0, newValues, 0, newSize);
            this.values = newValues;
        }
    }

    /**
     * Returns the {@code i}th value of this array.
     * @param i The index to get
     * @return The value
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    public float get(int i) throws IndexOutOfBoundsException {
        if (i < 0 || i >= size) {
            throw new IndexOutOfBoundsException("Invalid index " + i + " for size of " + size);
        }

        return values[i];
    }

    /**
     * Sets the {@code i}th value of this array.
     * @param i The index to set
     * @param value The value to set to
     * @throws IndexOutOfBoundsException When the index is out of bounds
     */
    public void set(int i, float value) throws IndexOutOfBoundsException {
        if (i < 0 || i >= size) {
            throw new IndexOutOfBoundsException("Invalid index " + i + " for size of " + size);
        }

        values[i] = value;
    }

    /**
     * Bulk sets the {@code i}th value of this array, with values starting at that position. If the given array overflows
     * the existing data, it will be appended. If {@code i == size}, meaning the operation would be pure append, this method
     * does not run - it instead delegates to {@link #append(float...)}.
     *
     * @param i The index to start setting
     * @param values The values to overwrite with
     * @throws IndexOutOfBoundsException When the index is invalid
     * @throws OutOfEmptyIndexException When the array is full
     */
    public void set(int i, float[] values) throws IndexOutOfBoundsException, OutOfEmptyIndexException {
        if (i < 0 || i > size) {
            throw new IndexOutOfBoundsException("Invalid index " + i + " for size of " + size);
        } else if (i == size) {
            append(values);
            return;
        }

        Objects.requireNonNull(values, "Input array cannot be null.");

        int payloadSize = values.length;
        int lastIndexToSet = i + payloadSize;

        boolean needsNominalResize = lastIndexToSet >= size;
        boolean needsActualResize = lastIndexToSet >= this.values.length;
        boolean needsResize = needsNominalResize || needsActualResize;

        if (needsResize) {
            if (Integer.MAX_VALUE - lastIndexToSet < this.values.length) {
                throw new OutOfEmptyIndexException("This TokenArray cannot take " + payloadSize + " more tokens. The available capacity is " + (Integer.MAX_VALUE - size));
            }

            if (needsActualResize) {
                resizeNominal(lastIndexToSet);
            } else {
                this.size = lastIndexToSet;
            }
        }

        System.arraycopy(values, 0, this.values, i, payloadSize);
    }

    /**
     * Appends to this array.
     * @param value The value to append
     * @throws OutOfEmptyIndexException When the array is full
     */
    public void append(float value) throws OutOfEmptyIndexException {
        if (size == Integer.MAX_VALUE) {
            throw new OutOfEmptyIndexException("This TokenArray is full - no more indexes are available.");
        }

        int index = size;
        int newSize = size + 1;

        if (newSize <= values.length) {
            values[index] = value;
            this.size = newSize;
        } else {
            resizeNominal(newSize);
            values[index] = value;
        }
    }

    /**
     * Appends multiple values to this array.
     * @param values The values to append
     * @throws OutOfEmptyIndexException When the array is full
     */
    public void append(float... values) throws OutOfEmptyIndexException {
        Objects.requireNonNull(values, "Cannot append TokenArray with null array");

        int payloadSize = values.length;

        if (Integer.MAX_VALUE - payloadSize < size) {
            throw new OutOfEmptyIndexException("This TokenArray cannot take " + payloadSize + " more tokens. The available capacity is " + (Integer.MAX_VALUE - size));
        }

        int indexToStart = size;
        int newSize = size + payloadSize;
        int newSizeActual = getBufferedNewSize(newSize);

        float[] newValues = new float[newSizeActual];
        System.arraycopy(this.values, 0, newValues, 0, size);
        System.arraycopy(values, 0, newValues, indexToStart, payloadSize);

        this.values = newValues;
        this.size = newSize;
    }

    /**
     * Appends another token array to this array.
     * @param array The array to append
     * @throws OutOfEmptyIndexException When the array is full
     */
    public void append(TokenArray array) throws OutOfEmptyIndexException {
        Objects.requireNonNull(array, "Cannot append TokenArray with null array");

        int payloadSize = array.size;

        if (Integer.MAX_VALUE - payloadSize < size) {
            throw new OutOfEmptyIndexException("This TokenArray cannot take " + payloadSize + " more tokens. The available capacity is " + (Integer.MAX_VALUE - size));
        }

        int indexToStart = size;
        int newSize = size + payloadSize;

        resizeNominal(newSize);

        System.arraycopy(array.values, 0, this.values, indexToStart, payloadSize);
    }

    /**
     * Returns a buffered new size.
     * @param minimumSize The minimum size
     * @return The buffered size
     */
    private int getBufferedNewSize(int minimumSize) {
        if (minimumSize >= FINAL_BLOCK_LIMIT) return Integer.MAX_VALUE;
        else if (minimumSize > BLOCK_SIZE * 10) return minimumSize + BLOCK_SIZE;
        else if (minimumSize > BLOCK_SIZE * 5) return minimumSize + HALF_BLOCK_SIZE;
        else return minimumSize + QUARTER_BLOCK_SIZE;
    }

    /**
     * Clears the array, but does not resize to zero. To force delete the array, call {@link #resizeActual(int)} to resize to zero.
     */
    public void clear() {
        size = 0;
    }

    /**
     * Returns the contents of this array.
     * @return The contents of this array
     */
    public float[] toArray() {
        return Arrays.copyOf(values, size);
    }

    /**
     * Returns a copy of the raw array underneath this TokenArray.
     * @return A copy of the raw untrimmed array
     */
    public float[] toArrayRaw() {
        return Arrays.copyOf(values, values.length);
    }

    /**
     * Trims all buffer and resizes this array to its exact size. Identical to calling {@link #resizeActual(int)} to resize to {@link #size()}.
     */
    public void trim() {
        resizeActual(size);
    }

    /**
     * Returns a real-time iterator, not thread-safe.
     * @return The iterator
     */
    @Override
    public Iterator<Float> iterator() {
        return new TokenArrayIterator(size, values);
    }

    /**
     * Real-time iterator.
     */
    private static final class TokenArrayIterator implements Iterator<Float> {
        private TokenArrayIterator(int size, float[] values) {
            this.size = size;
            this.referenceToArray = values;
            this.index = 0;
        }

        private int index;
        private final int size;
        private final float[] referenceToArray;


        @Override
        public boolean hasNext() {
            return index < size;
        }

        @Override
        public Float next() {
            return referenceToArray[index++];
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof TokenArray ta)) return false;
        return size == ta.size && Arrays.equals(values, 0, size, ta.values, 0, size);
    }

    @Override
    public int hashCode() {
        return Objects.hash(size, Arrays.hashCode(Arrays.copyOf(values, size)));
    }

    @Override
    public String toString() {
        return "TokenArray{" +
                "size=" + size +
                ", sizeActual=" + values.length +
                ", values=" + Arrays.toString(Arrays.copyOf(values, size)) +
                '}';
    }
}
