package team.trimark.javallm.type;

public interface TokenArrayCompatible extends Iterable<Float> {
    int size();
    float get(int i) throws IndexOutOfBoundsException;
    void set(int i, float value) throws IndexOutOfBoundsException;
    void set(int i, float[] values) throws IndexOutOfBoundsException;
    void clear();
    float[] toArray();
}
