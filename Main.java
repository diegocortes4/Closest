import java.util.ArrayList;
import java.util.List;

public class KNNTest {

  public static void main(String[] args) {
    
    // Create some sample data
    List<DataPoint> data = new ArrayList<>();
    data.add(new DataPoint(new double[]{1, 2}, 0)); 
    data.add(new DataPoint(new double[]{1.5, 2.5}, 0));
    data.add(new DataPoint(new double[]{3, 4}, 1));
    data.add(new DataPoint(new double[]{4, 4}, 1));
    
    // Create classifier and train
    KNNClassifier classifier = new KNNClassifier(data);
    
    // Make a prediction
    DataPoint input = new DataPoint(new double[]{2, 3}, 0);
    String prediction = classifier.predict(input, 3);
    
    // Print prediction
    System.out.println(prediction);

  }

}