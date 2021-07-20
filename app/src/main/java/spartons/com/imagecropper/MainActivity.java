package spartons.com.imagecropper;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.yalantis.ucrop.UCrop;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Objects;

import spartons.com.imagecropper.enums.ImagePickerEnum;
import spartons.com.imagecropper.listeners.IImagePickerLister;
import spartons.com.imagecropper.utils.FileUtils;
import spartons.com.imagecropper.utils.UiHelper;

public class MainActivity extends AppCompatActivity implements IImagePickerLister {

    private static final int CAMERA_ACTION_PICK_REQUEST_CODE = 610;
    private static final int PICK_IMAGE_GALLERY_REQUEST_CODE = 609;
    public static final int CAMERA_STORAGE_REQUEST_CODE = 611;
    public static final int ONLY_CAMERA_REQUEST_CODE = 612;
    public static final int ONLY_STORAGE_REQUEST_CODE = 613;

    private String currentPhotoPath = "";
    private UiHelper uiHelper = new UiHelper();
    private ImageView imageView;

    Button feedback;
    Button code, manual;
    TextView resistor;
    TextView bands;
    TextView title;

    String imageString="";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image_view);
        resistor = (TextView) findViewById(R.id.textView3);
        bands = (TextView) findViewById(R.id.textView);
        title = (TextView) findViewById(R.id.textView2);
        feedback = (Button) findViewById(R.id.feedbackbtn);
        code = (Button) findViewById(R.id.codebtn);
        manual = (Button) findViewById(R.id.manualbtn);

        resistor.setText("Resistor Value");
        title.setText("ResCalc");

        feedback.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openfeedbackform();
            }
        });

        code.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                opencode();
            }
        });

        manual.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openmanual();
            }
        });

        findViewById(R.id.capture_image_btn).setOnClickListener(v -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M)
                if (uiHelper.checkSelfPermissions(this))
                    uiHelper.showImagePickerDialog(this, this);
        });

    }

    //opens resistor code reference page
    public void opencode() {
        Intent intent = new Intent(this,resistorcode.class);
        startActivity(intent);
    }

    //opens feedback form pagr
    public void openfeedbackform() {
        Intent intent = new Intent(this,feedbackform.class);
        startActivity(intent);
    }

    //opens manual resistor value calculation page
    public void openmanual() {
        Intent intent = new Intent(this,manual.class);
        startActivity(intent);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_STORAGE_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED)
                uiHelper.showImagePickerDialog(this, this);
            else if (grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_DENIED) {
                uiHelper.toast(this, "ImageCropper needs Storage access in order to store your profile picture.");
                finish();
            } else if (grantResults[0] == PackageManager.PERMISSION_DENIED && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                uiHelper.toast(this, "ImageCropper needs Camera access in order to take profile picture.");
                finish();
            } else if (grantResults[0] == PackageManager.PERMISSION_DENIED && grantResults[1] == PackageManager.PERMISSION_DENIED) {
                uiHelper.toast(this, "ImageCropper needs Camera and Storage access in order to take profile picture.");
                finish();
            }
        } else if (requestCode == ONLY_CAMERA_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
                uiHelper.showImagePickerDialog(this, this);
            else {
                uiHelper.toast(this, "ImageCropper needs Camera access in order to take profile picture.");
                finish();
            }
        } else if (requestCode == ONLY_STORAGE_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
                uiHelper.showImagePickerDialog(this, this);
            else {
                uiHelper.toast(this, "ImageCropper needs Storage access in order to store your profile picture.");
                finish();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAMERA_ACTION_PICK_REQUEST_CODE && resultCode == RESULT_OK) {
            Uri uri = Uri.parse(currentPhotoPath);
            openCropActivity(uri, uri);
        } else if (requestCode == UCrop.REQUEST_CROP && resultCode == RESULT_OK) {
            if (data != null) {
                Uri uri = UCrop.getOutput(data);
                showImage(uri);
                Log.d("myTAG", "going to colour detect");
                //starts colour detection process after image croppiing
                colourdetection(uri);
            }
        } else if (requestCode == PICK_IMAGE_GALLERY_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            try {
                Uri sourceUri = data.getData();
                File file = getImageFile();
                Uri destinationUri = Uri.fromFile(file);
                openCropActivity(sourceUri, destinationUri);
            } catch (Exception e) {
                uiHelper.toast(this, "Please select another image");
            }
        }
    }

    private void openImagesDocument() {
        Intent pictureIntent = new Intent(Intent.ACTION_GET_CONTENT);
        pictureIntent.setType("image/*");
        pictureIntent.addCategory(Intent.CATEGORY_OPENABLE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            String[] mimeTypes = new String[]{"image/jpeg", "image/png"};
            pictureIntent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes);
        }
        startActivityForResult(Intent.createChooser(pictureIntent, "Select Picture"), PICK_IMAGE_GALLERY_REQUEST_CODE);
    }

    private void showImage(Uri imageUri) {
        try {
            File file;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.KITKAT) {
                file = FileUtils.getFile(this, imageUri);
            } else {
                file = new File(currentPhotoPath);
            }
            InputStream inputStream = new FileInputStream(file);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
            imageView.setImageBitmap(bitmap);
        } catch (Exception e) {
            uiHelper.toast(this, "Please select different profile picture.");
        }
    }

    @Override
    public void onOptionSelected(ImagePickerEnum imagePickerEnum) {
        if (imagePickerEnum == ImagePickerEnum.FROM_CAMERA)
            openCamera();
        else if (imagePickerEnum == ImagePickerEnum.FROM_GALLERY)
            openImagesDocument();
    }

    private void openCamera() {
        Intent pictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        File file;
        try {
            file = getImageFile(); // 1
        } catch (Exception e) {
            e.printStackTrace();
            uiHelper.toast(this, "Please take another image");
            return;
        }
        Uri uri;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) // 2
            uri = FileProvider.getUriForFile(this, BuildConfig.APPLICATION_ID.concat(".provider"), file);
        else
            uri = Uri.fromFile(file); // 3
        pictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, uri); // 4
        startActivityForResult(pictureIntent, CAMERA_ACTION_PICK_REQUEST_CODE);
    }

    private File getImageFile() throws IOException {
        String imageFileName = "JPEG_" + System.currentTimeMillis() + "_";
        File storageDir = new File(
                Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_DCIM
                ), "Camera"
        );
        System.out.println(storageDir.getAbsolutePath());
        if (storageDir.exists())
            System.out.println("File exists");
        else
            System.out.println("File not exists");
        File file = File.createTempFile(
                imageFileName, ".jpg", storageDir
        );
        currentPhotoPath = "file:" + file.getAbsolutePath();
        return file;
    }

    private void openCropActivity(Uri sourceUri, Uri destinationUri) {
        UCrop.Options options = new UCrop.Options();
        options.setCircleDimmedLayer(true);
        options.setCropFrameColor(ContextCompat.getColor(this, R.color.colorAccent));
        UCrop.of(sourceUri, destinationUri)
                .withMaxResultSize(100, 100)
                .withAspectRatio(10f, 3f)
                .start(this);
    }

    //get string from a bitmap of an image for input into python file
    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        //store in byte array
        byte[] imageBytes = baos.toByteArray();
        //finally encode to string
        String encodedImage = android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;
    }

    //entire colour detection pprocess
    private void colourdetection(Uri image_uri){
        Bitmap bitmap = null;
        try {
            //get bit map from image uri
            bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), image_uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        //Log.d("myTAG", "bitmap successfully stored");
        //pass string to a string variable
        imageString = getStringImage(bitmap);
        //Log.d("myTAG", "bitmap successfully stored");
        //start Chaquopy
        if(! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        final Python pyy = Python.getInstance();
        //find python file
        PyObject pyoo = pyy.getModule("ColourSeparation");
        try {
            //calls main function in python file and inputs imageString, returns a list
            List<PyObject> objj = pyoo.callAttr("main", imageString).asList();
            Log.d("myTAG", "objj is not null");
            String resvaluestr = objj.get(0).toString();
            String bandstr = objj.get(1).toString();
            resistor.setText(resvaluestr);
            bands.setText(bandstr);
        } catch(Exception e){
            Log.d("myTAG", "objj is null");
            //ImageView.setImageURI(image_uri);
            resistor.setText("Detection error! Try Again.");
            bands.setText("-");
        }
    }
}
