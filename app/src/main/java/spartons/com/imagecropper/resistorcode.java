package spartons.com.imagecropper;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class resistorcode extends AppCompatActivity {

    Button home;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_resistorcode);

        home = (Button) findViewById(R.id.btn);

        home.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openhome();
            }
        });
    }

    public void openhome() {
        Intent intent = new Intent(this,MainActivity.class);
        startActivity(intent);
    }
}