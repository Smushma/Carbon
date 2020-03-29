package org.tensorflow.lite.examples.detection;

import androidx.appcompat.app.AppCompatActivity;



import android.app.ActivityOptions;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.examples.detection.R;

public class SplashScreenActivity extends AppCompatActivity {
    // How long the splash screen lasts
    private static int SPLASH_SCREEN = 5000;

    private Animation topAnim, bottomAnim;
    private ImageView img;
    private TextView logo, slogan;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_splash_screen);

        // Animations
        topAnim = AnimationUtils.loadAnimation(this, R.anim.top_animation);
        bottomAnim = AnimationUtils.loadAnimation(this, R.anim.bottom_animation);

        // Hooks
        img = findViewById(R.id.logo);
        logo = findViewById(R.id.tvSplashLogo);
        slogan = findViewById(R.id.tvSplashSlogan);

        // Set animation
        img.setAnimation(topAnim);
        logo.setAnimation(bottomAnim);
        slogan.setAnimation(bottomAnim);

        // Go to login after splash
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent login = new Intent(SplashScreenActivity.this, org.tensorflow.lite.examples.detection.DetectorActivity.class);
                // Don't allow this activity to be on the stack
                login.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_CLEAR_TASK);
                startActivity(login);

                finish();
            }
        }, SPLASH_SCREEN);
    }
}
