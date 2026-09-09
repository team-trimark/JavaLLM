package team.trimark.javallm.type;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Real-time iterator.
 */
final class TokenArrayLiveIterator implements Iterator<Float> {
    /**
     * Creates a new iterator over the first {@code size} entries of the provided array.
     * @param size The number of readable entries
     * @param values The array to iterate over
     */
    TokenArrayLiveIterator(int size, float[] values) {
        this.size = size;
        this.referenceToArray = values;
        this.index = 0;
    }

    /**
     * The number of readable entries.
     */
    private final int size;

    /**
     * The array being iterated over.
     */
    private final float[] referenceToArray;

    /**
     * The index of the next entry to return.
     */
    private int index;

    /**
     * Checks whether there is another entry to read.
     * @return {@code true} if there is at least one entry left
     */
    @Override
    public boolean hasNext() {
        return index < size;
    }

    /**
     * Returns the next entry of this iterator.
     * @return The next entry
     * @throws NoSuchElementException When there are no entries left
     */
    @Override
    public Float next() {
        if (index >= size) {
            throw new NoSuchElementException("No more tokens are available.");
        }

        return referenceToArray[index++];
    }
}
