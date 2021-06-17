package com.example.camera;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.ContentValues;
//import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_CAPTURE_CODE = 1001;
    private static final int PERMISSION_CODE = 1000;

    Button mCaptureBtn;
    ImageView mImageView;
    TextView resistor;
    TextView bands;
    TextView title;

    Uri image_uri;

    BitmapDrawable drawable;
    Bitmap bitmap;
    String imageString="";
    String maskString="";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.image_view);
        mCaptureBtn = findViewById(R.id.capture_image_btn);
        resistor = (TextView) findViewById(R.id.textView3);
        bands = (TextView) findViewById(R.id.textView);
        title = (TextView) findViewById(R.id.textView2);

        resistor.setText("Resistor Value");
        title.setText("ResCalc");
        //button click
        mCaptureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick (View v){
                //if system os is >= marshmellow, request runtime permission
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) ==
                            PackageManager.PERMISSION_DENIED || checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
                        //permission not enabled,request it
                        String[] permission = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
                        //show popup to request permissions
                        requestPermissions(permission, PERMISSION_CODE);
                    } else {
                        //permission already granted
                        openCamera();
                    }
                } else {
                    //system < marshmellow
                    openCamera();
                }
            }
        });
    }

    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From the Camera");
        image_uri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        //Camera intent
        Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, image_uri);
        startActivityForResult(cameraIntent, IMAGE_CAPTURE_CODE);
        //Log.d("myTAG", "bitmap successfully stored");
    }

    //handling permission result
    @Override
    public void onRequestPermissionsResult (int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        //this method is called, when user presses Allow or Deny from Permission Request Popup
        switch (requestCode) {
            case PERMISSION_CODE: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    //permission from popup was granted
                    openCamera();
                }
                else {
                    //permission from popup was denied
                    Toast.makeText(this, "Permission denied...", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    @Override
    protected void onActivityResult (int requestCode, int resultCode, Intent data) {
        //called when image was captured from camera
        //super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            //set the image captured to our ImageView
            //mImageView.setImageURI(image_uri);
            //Log.d("myTAG", "bitmap successfully stored");
            if(! Python.isStarted()) {
                Python.start(new AndroidPlatform(this));
            }
            final Python py = Python.getInstance();

            Bitmap bitmap = null;
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), image_uri);
                //mImageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
            //Log.d("myTAG", "bitmap successfully stored");
            imageString = getStringImage(bitmap);
            //now pass image string to python script
            //Log.d("myTAG", "bitmap successfully stored");
            PyObject pyo = py.getModule("samples.resistor.ResistorDetection");
            //Log.d("myTAG", "bitmap successfully stored");
            //call main method of script and pass image string as parameter
            PyObject obj = pyo.callAttr("main",imageString);
            Log.d("myTAG", "after python file");

            if (Objects.isNull(obj) ){
                Log.d("myTAG", "obj is null");
                mImageView.setImageURI(image_uri);
                resistor.setText("No detection! Try Again.");
                bands.setText("-");
            } else {
                Log.d("myTAG", "obj is not null");


                //obj will contain return value  ie image string
                String str = obj.toString();
                //convert to byte array
                byte dataa[] = android.util.Base64.decode(str, Base64.DEFAULT);
                //convert to bitmap
                Bitmap bmp = BitmapFactory.decodeByteArray(dataa, 0, dataa.length);

                int width = bmp.getWidth();
                int height = bmp.getHeight();
                Log.d("ADebugTag", "Value: " + Float.toString(width));
                Log.d("ADebugTag", "Value: " + Float.toString(height));

                //rotate bitmap -90 degrees
                Bitmap rotated = rotateBitmap(bmp, 90);

                //set it to imageview2
                mImageView.setImageBitmap(rotated);

                final Python pyy = Python.getInstance();
                maskString = getStringImage(rotated);
                PyObject pyoo = pyy.getModule("samples.resistor.ColourDetection.ColourSeparation");
                List<PyObject> objj = pyoo.callAttr("main", maskString).asList();
                if (Objects.isNull(objj)){
                    Log.d("myTAG", "objj is null");
                    mImageView.setImageURI(image_uri);
                    resistor.setText("Detection error! Try Again.");
                    bands.setText("-");
                }else{
                    Log.d("myTAG", "objj is not null");
                    String strr = objj.get(0).toString();
                    String strrr = objj.get(1).toString();
                    resistor.setText(strr);
                    bands.setText(strrr);
                }
            }

        }
    }

    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        //store in byte array
        byte[] imageBytes = baos.toByteArray();
        //finally encode to string
        String encodedImage = android.util.Base64.encodeToString(imageBytes,Base64.DEFAULT);
        return encodedImage;
    }

    public Bitmap rotateBitmap(Bitmap original, float degrees) {
        Matrix matrix = new Matrix();
        matrix.preRotate(degrees);
        Bitmap rotatedBitmap = Bitmap.createBitmap(original, 0, 0, original.getWidth(), original.getHeight(), matrix, true);
        original.recycle();
        return rotatedBitmap;
    }

}