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
    int rows() {
        return height;
    }

    /**
     * The number of columns.
     * @return The number of columns
     */
    int columns() {
        return width;
    }

    /**
     * Returns the surface area.
     * @return The surface area
     */
    int area() {
        return width * height;
    }

    @Override
    public int hashCode() {
        return Objects.hash(width, height);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Dimension d)) return false;
        return width == d.width && height == d.height;
    }

    @Override
    public String toString() {
        return "Dimension{" +
                "width=" + width +
                ", height=" + height +
                '}';
    }
}
