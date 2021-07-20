package spartons.com.imagecropper;

import java.util.HashMap;
import java.util.Map;

public class ToleranceCodeConversion {

    private static Map<String, Double> toleranceValues = new HashMap<>();

    public static final Integer UNKNOWN_COLOR_VALUE = -1;

    static {
        toleranceValues.put("Brown", 0.01);
        toleranceValues.put("Red", 0.02);
        toleranceValues.put("Orange", 0.0005);
        toleranceValues.put("Yellow", 0.0002);
        toleranceValues.put("Green", 0.005);
        toleranceValues.put("Blue", 0.0025);
        toleranceValues.put("Violet", 0.001);
        toleranceValues.put("Grey", 0.0001);
        toleranceValues.put("Gold", 0.05);
        toleranceValues.put("Silver", 0.1);
    }

    public static double getValueForTolerance(String colourName) {
        if (toleranceValues.containsKey(colourName)) {
            return toleranceValues.get(colourName);
        }

        return UNKNOWN_COLOR_VALUE;
    }

}
