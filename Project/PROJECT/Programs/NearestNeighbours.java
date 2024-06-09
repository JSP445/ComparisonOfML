import java.util.ArrayList;
import java.util.List;

class Point {

    // Creates a 2D point with x and y coordinates
    // Has a method to calculate the Euclidean distance between two points.

    double x;
    double y;

    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double distanceTo(Point other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
}

public class NearestNeighbours {

    // Nearest Neighbour method takes a target point and a list of points and
    // returns the nearest neighbour using Euclidean formula.

    public static Point findNearestNeighbour(Point target, List<Point> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("No points provided");
        }

        Point nearestNeighbour = points.get(0);
        double minDistance = target.distanceTo(nearestNeighbour);

        for (Point point : points) {
            double distance = target.distanceTo(point);
            if (distance < minDistance) {
                minDistance = distance;
                nearestNeighbour = point;
            }
        }

        return nearestNeighbour;
    }

    public static void main(String[] args) {
        List<Point> points = new ArrayList<>();
        points.add(new Point(0, 0));
        points.add(new Point(3, 4));
        points.add(new Point(1, 1));
        points.add(new Point(10, 10));

        Point target = new Point(0, 0);
        Point nearestNeighbour = findNearestNeighbour(target, points);

        System.out.println("Nearest Neighbour to (" + target.x + ", " + target.y + ") is at (" + nearestNeighbour.x
                + ", " + nearestNeighbour.y + ")");

    }
}