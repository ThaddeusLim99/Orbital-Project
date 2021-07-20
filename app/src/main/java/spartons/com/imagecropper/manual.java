package spartons.com.imagecropper;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class manual extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    RadioGroup radioGroup;
    RadioButton radioButton;
    TextView textView, resistorValue;
    Button calcBtn;

    private static final String[] bandColours = {"", "Black", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White"};
    private static final String[] firstBandColours = {"", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White"};
    private static final String[] multiplierColours = {"", "Black", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "White", "Gold", "Silver"};
    private static final String[] toleranceColours = {"", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey", "Gold", "Silver"};
    private static final String[] PPMColours = {"", "Black", "Brown", "Red", "Orange", "Yellow", "Green", "Blue", "Violet", "Grey"};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_3band);

        radioGroup = findViewById(R.id.radioGroup);
        textView = findViewById(R.id.Title);
        resistorValue = (TextView) findViewById(R.id.resistorValue);
        calcBtn = (Button) findViewById(R.id.calculate);

        resistorValue.setText("Resistor Value");

        //Getting the instance of Spinner and applying OnItemSelectedListener on it
        Spinner spin1 = (Spinner) findViewById(R.id.spinner1); //band1
        Spinner spin2 = (Spinner) findViewById(R.id.spinner2); //band2
        Spinner spin4 = (Spinner) findViewById(R.id.spinner4); //multiplier


        //Creating the ArrayAdapter instance having the country list
        ArrayAdapter aa = new ArrayAdapter(this, android.R.layout.simple_spinner_item, bandColours);
        aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayAdapter aa2 = new ArrayAdapter(this, android.R.layout.simple_spinner_item, firstBandColours);
        aa2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayAdapter aa3 = new ArrayAdapter(this, android.R.layout.simple_spinner_item, multiplierColours);
        aa3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        //Setting the ArrayAdapter data on the Spinner
        spin1.setOnItemSelectedListener(this);
        spin1.setAdapter(aa2);

        spin2.setOnItemSelectedListener(this);
        spin2.setAdapter(aa);

        spin4.setOnItemSelectedListener(this);
        spin4.setAdapter(aa3);

        calcBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                threeBandCalc(spin1.getSelectedItem().toString(), spin2.getSelectedItem().toString(), spin4.getSelectedItem().toString());
            }
        });
    }
    public void threeBandCalc(String first, String second, String m)  {
        double res = (ColourCodeConversion.getValueForBand(first) * 10 + ColourCodeConversion.getValueForBand(second)) * Math.pow(10, ColourCodeConversion.getValueForBand(m));
        String resistance = res + " Ohms";
        if (res >= 1000000) {
            res /= 1000000;
            resistance = res + " MOhms";
        }
        if (res >= 1000)  {
            res /= 1000;
            resistance = res + " kOhms";
        }
        String result = resistance + " Tolerance: 20%";
        resistorValue.setText(result);
    }

    public void fourBandCalc(String first, String second, String m, String tol)  {
        double res = (ColourCodeConversion.getValueForBand(first) * 10 + ColourCodeConversion.getValueForBand(second)) * Math.pow(10, ColourCodeConversion.getValueForBand(m));
        String resistance = res + " Ohms";
        if (res >= 1000000) {
            res /= 1000000;
            resistance = res + " MOhms";
        }
        if (res >= 1000)  {
            res /= 1000;
            resistance = res + " kOhms";
        }
        String result = resistance + " Tolerance: " + ToleranceCodeConversion.getValueForTolerance(tol)*100 + "%";
        resistorValue.setText(result);
    }

    public void fiveBandCalc(String first, String second, String third, String m, String tol)  {
        double res = (ColourCodeConversion.getValueForBand(first) * 100 + ColourCodeConversion.getValueForBand(second) * 10 + ColourCodeConversion.getValueForBand(third)) * Math.pow(10, ColourCodeConversion.getValueForBand(m));
        String resistance = res + " Ohms";
        if (res >= 1000000) {
            res /= 1000000;
            resistance = res + " MOhms";
        }
        if (res >= 1000)  {
            res /= 1000;
            resistance = res + " kOhms";
        }
        String result = resistance + " Tolerance: " + ToleranceCodeConversion.getValueForTolerance(tol)*100 + "%";
        resistorValue.setText(result);
    }

    public void sixBandCalc(String first, String second, String third, String m, String tol, String PPM)  {
        double res = (ColourCodeConversion.getValueForBand(first) * 100 + ColourCodeConversion.getValueForBand(second) * 10 + ColourCodeConversion.getValueForBand(third)) * Math.pow(10, ColourCodeConversion.getValueForBand(m));
        String resistance = res + " Ohms";
        if (res >= 1000000) {
            res /= 1000000;
            resistance = res + " MOhms";
        }
        if (res >= 1000)  {
            res /= 1000;
            resistance = res + " kOhms";
        }
        String result = resistance + " Tolerance: " + ToleranceCodeConversion.getValueForTolerance(tol)*100 + "%" + " PPM: " + PPMCode.getValueForPPM(PPM);
        resistorValue.setText(result);
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
    }

    public void onNothingSelected(AdapterView<?> parent) {
        // TODO Auto-generated method stub
    }

    public void checkButton(View v) {
        int radioId = radioGroup.getCheckedRadioButtonId();

        radioButton = findViewById(radioId);

        //Toast.makeText(this,"Chose" + radioButton.getText(),Toast.LENGTH_SHORT).show();

        //TODO: Change the number of bands to fill in according to the number chosen
        switch (v.getId())    {
            case R.id.threeBandRadioBtn:
                setContentView(R.layout.activity_3band);

                radioGroup = findViewById(R.id.radioGroup);
                textView = findViewById(R.id.Title);
                resistorValue = (TextView) findViewById(R.id.resistorValue);
                calcBtn = (Button) findViewById(R.id.calculate);

                resistorValue.setText("Resistor Value");

                //Getting the instance of Spinner and applying OnItemSelectedListener on it
                Spinner spin1 = (Spinner) findViewById(R.id.spinner1); //band1
                Spinner spin2 = (Spinner) findViewById(R.id.spinner2); //band2
                Spinner spin4 = (Spinner) findViewById(R.id.spinner4); //multiplier

                //Creating the ArrayAdapter instance having the country list
                ArrayAdapter aa = new ArrayAdapter(this,android.R.layout.simple_spinner_item,bandColours);
                aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                ArrayAdapter aa2 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,firstBandColours);
                aa2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                ArrayAdapter aa3 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,multiplierColours);
                aa3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                //Setting the ArrayAdapter data on the Spinner
                spin1.setOnItemSelectedListener(this);
                spin1.setAdapter(aa2);

                spin2.setOnItemSelectedListener(this);
                spin2.setAdapter(aa);

                spin4.setOnItemSelectedListener(this);
                spin4.setAdapter(aa3);

                calcBtn.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        threeBandCalc(spin1.getSelectedItem().toString(), spin2.getSelectedItem().toString(), spin4.getSelectedItem().toString());
                    }
                });
                break;
            case R.id.fourBandRadioBtn:
                setContentView(R.layout.activity_4band);

                radioGroup = findViewById(R.id.radioGroup);
                textView = findViewById(R.id.Title);
                resistorValue = (TextView) findViewById(R.id.resistorValue);
                calcBtn = (Button) findViewById(R.id.calculate);

                resistorValue.setText("Resistor Value");

                //Getting the instance of Spinner and applying OnItemSelectedListener on it
                spin1 = (Spinner) findViewById(R.id.spinner1); //band1
                spin2 = (Spinner) findViewById(R.id.spinner2); //band2
                spin4 = (Spinner) findViewById(R.id.spinner4); //multiplier
                Spinner spin5 = (Spinner) findViewById(R.id.spinner5); //tolerance

                aa = new ArrayAdapter(this,android.R.layout.simple_spinner_item,bandColours);
                aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa2 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,firstBandColours);
                aa2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa3 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,multiplierColours);
                aa3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                ArrayAdapter aa4 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,toleranceColours);
                aa4.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                //Setting the ArrayAdapter data on the Spinner
                spin1.setOnItemSelectedListener(this);
                spin1.setAdapter(aa2);

                spin2.setOnItemSelectedListener(this);
                spin2.setAdapter(aa);

                spin4.setOnItemSelectedListener(this);
                spin4.setAdapter(aa3);

                spin5.setOnItemSelectedListener(this);
                spin5.setAdapter(aa4);

                calcBtn.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        fourBandCalc(spin1.getSelectedItem().toString(), spin2.getSelectedItem().toString(), spin4.getSelectedItem().toString(), spin5.getSelectedItem().toString());
                    }
                });
                break;
            case R.id.fiveBandRadioBtn:
                setContentView(R.layout.activity_5band);

                radioGroup = findViewById(R.id.radioGroup);
                textView = findViewById(R.id.Title);
                resistorValue = (TextView) findViewById(R.id.resistorValue);
                calcBtn = (Button) findViewById(R.id.calculate);

                resistorValue.setText("Resistor Value");

                //Getting the instance of Spinner and applying OnItemSelectedListener on it
                spin1 = (Spinner) findViewById(R.id.spinner1); //band1
                spin2 = (Spinner) findViewById(R.id.spinner2); //band2
                Spinner spin3 = (Spinner) findViewById(R.id.spinner3); //band3
                spin4 = (Spinner) findViewById(R.id.spinner4); //multiplier
                spin5 = (Spinner) findViewById(R.id.spinner5); //tolerance

                //Creating the ArrayAdapter instance having the country list
                aa = new ArrayAdapter(this,android.R.layout.simple_spinner_item,bandColours);
                aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa2 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,firstBandColours);
                aa2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa3 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,multiplierColours);
                aa3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa4 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,toleranceColours);
                aa4.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                //Setting the ArrayAdapter data on the Spinner
                spin1.setOnItemSelectedListener(this);
                spin1.setAdapter(aa2);

                spin2.setOnItemSelectedListener(this);
                spin2.setAdapter(aa);

                spin3.setOnItemSelectedListener(this);
                spin3.setAdapter(aa);

                spin4.setOnItemSelectedListener(this);
                spin4.setAdapter(aa3);

                spin5.setOnItemSelectedListener(this);
                spin5.setAdapter(aa4);

                calcBtn.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        fiveBandCalc(spin1.getSelectedItem().toString(), spin2.getSelectedItem().toString(), spin3.getSelectedItem().toString(), spin4.getSelectedItem().toString(), spin5.getSelectedItem().toString());
                    }
                });
                break;
            case R.id.sixBandRadioBtn:
                setContentView(R.layout.activity_manual);

                radioGroup = findViewById(R.id.radioGroup);
                textView = findViewById(R.id.Title);
                resistorValue = (TextView) findViewById(R.id.resistorValue);
                calcBtn = (Button) findViewById(R.id.calculate);

                resistorValue.setText("Resistor Value");

                //Getting the instance of Spinner and applying OnItemSelectedListener on it
                spin1 = (Spinner) findViewById(R.id.spinner1); //band1
                spin2 = (Spinner) findViewById(R.id.spinner2); //band2
                spin3 = (Spinner) findViewById(R.id.spinner3); //band3
                spin4 = (Spinner) findViewById(R.id.spinner4); //multiplier
                spin5 = (Spinner) findViewById(R.id.spinner5); //tolerance
                Spinner spin6 = (Spinner) findViewById(R.id.spinner6); //tolerance

                //Creating the ArrayAdapter instance having the country list
                aa = new ArrayAdapter(this,android.R.layout.simple_spinner_item,bandColours);
                aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa2 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,firstBandColours);
                aa2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa3 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,multiplierColours);
                aa3.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                aa4 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,toleranceColours);
                aa4.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

                ArrayAdapter aa5 = new ArrayAdapter(this,android.R.layout.simple_spinner_item,PPMColours);
                aa.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                //Setting the ArrayAdapter data on the Spinner
                spin1.setOnItemSelectedListener(this);
                spin1.setAdapter(aa2);

                spin2.setOnItemSelectedListener(this);
                spin2.setAdapter(aa);

                spin3.setOnItemSelectedListener(this);
                spin3.setAdapter(aa);

                spin4.setOnItemSelectedListener(this);
                spin4.setAdapter(aa3);

                spin5.setOnItemSelectedListener(this);
                spin5.setAdapter(aa4);

                spin6.setOnItemSelectedListener(this);
                spin6.setAdapter(aa5);

                calcBtn.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        sixBandCalc(spin1.getSelectedItem().toString(), spin2.getSelectedItem().toString(), spin3.getSelectedItem().toString(), spin4.getSelectedItem().toString(), spin5.getSelectedItem().toString(), spin6.getSelectedItem().toString());
                    }
                });
                break;
        }
    }
}