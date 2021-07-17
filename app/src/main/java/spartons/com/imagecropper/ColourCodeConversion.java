package spartons.com.imagecropper;

import java.util.HashMap;
import java.util.Map;

public class ColourCodeConversion {

    private static Map<String, Integer> colourValues = new HashMap<>();

    public static final Integer UNKNOWN_COLOR_VALUE = -1;

    static {
        colourValues.put("Black", 0);
        colourValues.put("Brown", 1);
        colourValues.put("Red", 2);
        colourValues.put("Orange", 3);
        colourValues.put("Yellow", 4);
        colourValues.put("Green", 5);
        colourValues.put("Blue", 6);
        colourValues.put("Violet", 7);
        colourValues.put("Grey", 8);
        colourValues.put("White", 9);
        colourValues.put("Gold", -1);
        colourValues.put("Silver", -2);
    }

    public static int getValueForBand(String colourName) {
        if (colourValues.containsKey(colourName)) {
            return colourValues.get(colourName);
        }

        return UNKNOWN_COLOR_VALUE;
    }

}
