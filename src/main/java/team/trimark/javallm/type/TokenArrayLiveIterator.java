package team.trimark.javallm.type;

import java.util.Iterator;

/**
 * Real-time iterator.
 */
final class TokenArrayLiveIterator implements Iterator<Float> {
    TokenArrayLiveIterator(int size, float[] values) {
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
