import java.util.List;
import java.util.PriorityQueue;

public class KNNClassifier {

  private List<DataPoint> trainingData;

  public KNNClassifier(List<DataPoint> trainingData) {
    this.trainingData = trainingData;
  }

  public String predict(DataPoint input, int k) {

    // Find k closest data points
    PriorityQueue<DataPoint> kClosest = findKClosest(input, k);

    // Get majority class of k neighbors
    return getMajorityClass(kClosest);
  }

  private PriorityQueue<DataPoint> findKClosest(DataPoint input, int k) {
    
    PriorityQueue<DataPoint> closest = new PriorityQueue<>(
      (a, b) -> Double.compare(distance(a, input), distance(b, input))
    );

    // Add all training data to priority queue
    for (DataPoint p : trainingData) {
      closest.add(p);
    }

    // Keep only k closest
    while (closest.size() > k) {
      closest.poll();
    }

    return closest;
  }

  private double distance(DataPoint a, DataPoint b) {
    // Euclidian distance
    double diff = 0;
    for (int i = 0; i < a.features.length; i++) {
      diff += Math.pow(a.features[i] - b.features[i], 2);
    }
    return Math.sqrt(diff);
  }

  private String getMajorityClass(PriorityQueue<DataPoint> kClosest) {
    int[] counts = new int[2];
    while (!kClosest.isEmpty()) {
      DataPoint p = kClosest.poll(); 
      counts[p.label]++; 
    }
    return counts[0] > counts[1] ? "0" : "1";
  }

}

class DataPoint {
  double[] features;
  int label; // 0 or 1
}