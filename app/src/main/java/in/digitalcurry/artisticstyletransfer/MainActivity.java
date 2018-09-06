package in.digitalcurry.artisticstyletransfer;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

public class MainActivity extends AppCompatActivity {

    //region ----- Instance Variables -----

    private TensorFlowInferenceInterface inferenceInterface;

    private static final String MODEL_FILE = "file:///android_asset/stylize_quantized.pb";

    private static final String INPUT_NODE = "input";
    private static final String STYLE_NODE = "style_num";
    private static final String OUTPUT_NODE = "transformer/expand/conv3/conv/Sigmoid";

    private int selectedStyle = 0;

    private Uri fileUri = null;
    private int OPEN_CAMERA_FOR_CAPTURE = 0x1;

    private static final int NUM_STYLES = 26;
    private float[] styleVals = new float[NUM_STYLES];

    private int desiredSize = 256;
    private int[] intValues = new int[desiredSize * desiredSize];

    private float[] floatValues = new float[desiredSize * desiredSize * 3];;

    //endregion

    //region ----- Bind Elements -----

    @BindView(R.id.camera_image)
    ImageView cameraImageView;

    //endregion

    //region ----- OnClick Listeners -----

    @OnClick({R.id.style_1, R.id.style_2, R.id.style_3})
    public void onStyleBtnClicked(View view) {

        styleVals = new float[NUM_STYLES];

        switch (view.getId()) {
            case R.id.style_1:
                selectedStyle = 1;
                styleVals[11] = 1.0f;
                break;
            case R.id.style_2:
                selectedStyle = 2;
                styleVals[19] = 1.0f;
                break;
            case R.id.style_3:
                selectedStyle = 3;
                styleVals[24] = 1.0f;
                break;
        }
    }

    @OnClick(R.id.button)
    public void onApplyBtnClicked(View view) {

        // Apply the style to the camera image
        stylizeImage(getScaledBitmap(fileUri));
    }

    //endregion

    //region ----- Activity Related Methods -----

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ButterKnife.bind(this);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {

        MenuInflater menuInflater = getMenuInflater();
        menuInflater.inflate(R.menu.main_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        if (item.getItemId() == R.id.camera_image) {

            File imageFile = new File(android.os.Environment.getExternalStorageDirectory(), "temp.jpg");

            fileUri = Uri.fromFile(imageFile);

            Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");
            intent.putExtra(MediaStore.EXTRA_OUTPUT, fileUri);
            startActivityForResult(intent, OPEN_CAMERA_FOR_CAPTURE);

        }

        return true;
    }

    //endregion

    //region ----- Tensorflow Related Methods -----

    private void stylizeImage(Bitmap bitmap) {

        cameraImageView.setImageBitmap(bitmap);

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }
        // Copy the input data into TensorFlow.
        inferenceInterface.feed(INPUT_NODE, floatValues, 1, bitmap.getWidth(), bitmap.getHeight(), 3);
        inferenceInterface.feed(STYLE_NODE, styleVals, NUM_STYLES);

        // Execute the output node's dependency sub-graph.
        inferenceInterface.run(new String[] {OUTPUT_NODE}, false);

        // Copy the data from TensorFlow back into our array.
        inferenceInterface.fetch(OUTPUT_NODE, floatValues);

        for (int i = 0; i < intValues.length; ++i) {
            intValues[i] =
                    0xFF000000
                            | (((int) (floatValues[i * 3] * 255)) << 16)
                            | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                            | ((int) (floatValues[i * 3 + 2] * 255));
        }

        bitmap.setPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        cameraImageView.setImageBitmap(bitmap);
    }

    //endregion

    //region ----- Helper Methods -----

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == OPEN_CAMERA_FOR_CAPTURE && resultCode == Activity.RESULT_OK) {

            try {

                cameraImageView.setImageBitmap(getScaledBitmap(fileUri));
            } catch (NullPointerException e) {
                e.printStackTrace();
            }
        }
    }

    public Bitmap getScaledBitmap(Uri fileUri) {

        Bitmap scaledPhoto = null;

        try {

            // Get the dimensions of the bitmap
            BitmapFactory.Options bmOptions = new BitmapFactory.Options();
            bmOptions.inJustDecodeBounds = true;

            Bitmap bitmap = BitmapFactory.decodeFile(fileUri.getPath());

            ExifInterface ei = new ExifInterface(fileUri.getPath());
            int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

            // if orientation is 6 or 3 then the photo taken is portrait
            switch(orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    bitmap = rotateImage(bitmap, 90);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    bitmap = rotateImage(bitmap, 180);
                    break;
            }

            bmOptions.inJustDecodeBounds = false;
            bmOptions.inPurgeable = true;

            // Scaling down the original image taken from camera
            scaledPhoto = Bitmap.createScaledBitmap(bitmap, desiredSize, desiredSize, false);

        }
        catch (Exception ex) {

        }

        return scaledPhoto;
    }

    public Bitmap rotateImage(Bitmap source, float angle) {
        Bitmap retVal;

        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        retVal = Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);

        return retVal;
    }

    //endregion
}
