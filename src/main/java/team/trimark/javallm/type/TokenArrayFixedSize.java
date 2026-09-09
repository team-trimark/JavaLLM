package team.trimark.javallm.type;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;

public class TokenArrayFixedSize implements TokenArrayCompatible {
    public final float[] values;

    public TokenArrayFixedSize(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("TokenArrayFixedSize size cannot be negative.");
        }

        this.values = new float[size];
    }

    public static TokenArrayFixedSize copyOf(float... values) {
        Objects.requireNonNull(values, "Array cannot be null.");
        return new TokenArrayFixedSize(Arrays.copyOf(values, values.length));
    }

    public static TokenArrayFixedSize refOf(float[] values) {
        Objects.requireNonNull(values, "Array cannot be null.");
        return new TokenArrayFixedSize(values);
    }

    private TokenArrayFixedSize(float[] values) {
        this.values = Objects.requireNonNull(values);
    }

    public TokenArrayFixedSize(TokenArrayCompatible tac) {
        this(Objects.requireNonNull(tac, "Source TokenArray cannot be null.").size());

        if (tac instanceof TokenArrayFixedSize tafs) {
            System.arraycopy(tafs.values, 0, this.values, 0, values.length);
        } else {
            System.arraycopy(tac.toArray(), 0, this.values, 0, values.length);
        }
    }

    @Override
    public Iterator<Float> iterator() {
        return new TokenArrayLiveIterator(values.length, values);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float get(int i) throws IndexOutOfBoundsException {
        return values[i];
    }

    @Override
    public void set(int i, float value) throws IndexOutOfBoundsException {
        values[i] = value;
    }

    @Override
    public void set(int i, float[] values) throws IndexOutOfBoundsException {
        if (i < 0 || i >= values.length) {
            throw new IndexOutOfBoundsException("Index out of bounds.");
        }

        int payloadSize = Objects.requireNonNull(values, "Cannot be null.").length;
        int lastIndexAfter = i + payloadSize;
        if (lastIndexAfter >= values.length) {
            throw new IndexOutOfBoundsException("Array is to large to fit.");
        }

        System.arraycopy(values, 0, this.values, i, payloadSize);
    }

    @Override
    public void clear() {
        Arrays.fill(values, 0f);
    }

    @Override
    public float[] toArray() {
        return Arrays.copyOf(values, values.length);
    }
}
