package spartons.com.imagecropper;

import java.util.HashMap;
import java.util.Map;

public class PPMCode {

    private static Map<String, Integer> PPMValues = new HashMap<>();

    public static final Integer UNKNOWN_COLOR_VALUE = -1;

    static {
        PPMValues.put("Black", 250);
        PPMValues.put("Brown", 100);
        PPMValues.put("Red", 50);
        PPMValues.put("Orange", 15);
        PPMValues.put("Yellow", 25);
        PPMValues.put("Green", 20);
        PPMValues.put("Blue", 10);
        PPMValues.put("Violet", 5);
        PPMValues.put("Grey", 1);
    }

    public static double getValueForPPM(String colourName) {
        if (PPMValues.containsKey(colourName)) {
            return PPMValues.get(colourName);
        }

        return UNKNOWN_COLOR_VALUE;
    }

}
