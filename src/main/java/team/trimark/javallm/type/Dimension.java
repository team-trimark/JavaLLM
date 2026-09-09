package team.trimark.javallm.type;

import java.util.Objects;

/**
 * The dimensions of a two-dimensional datatype.
 * @param width The width
 * @param height The height
 */
public record Dimension(
        int width,
        int height
) {
    /**
     * Rows-columns.
     * @param rows r
     * @param columns c
     * @return The dimensions
     */
    public static Dimension rowCols(int rows, int columns) {
        return new Dimension(columns, rows);
    }

    /**
     * Columns-rows.
     * @param columns c
     * @param rows r
     * @return The dimensions
     */
    public static Dimension colRows(int columns, int rows) {
        return new Dimension(columns, rows);
    }

    /**
     * The number of rows.
     * @return The number of rows
     */
    public int rows() {
        return height;
    }

    /**
     * The number of columns.
     * @return The number of columns
     */
    public int columns() {
        return width;
    }

    /**
     * Returns the surface area. Note that this overflows silently for very large dimensions;
     * a negative result indicates an overflow.
     * @return The surface area
     */
    public int area() {
        return width * height;
    }

    /**
     * Returns the hash code of these dimensions.
     * @return The hash code of these dimensions
     */
    @Override
    public int hashCode() {
        return Objects.hash(width, height);
    }

    /**
     * Checks for equality with another object.
     * @param obj The object to compare to
     * @return {@code true} if the other object has the same width and height
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Dimension d)) return false;
        return width == d.width && height == d.height;
    }

    /**
     * Serializes these dimensions into a string.
     * @return The string representation of these dimensions
     */
    @Override
    public String toString() {
        return "Dimension{" +
                "width=" + width +
                ", height=" + height +
                '}';
    }
}
