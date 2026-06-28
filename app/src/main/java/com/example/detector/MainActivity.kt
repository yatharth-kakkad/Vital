package com.example.detector

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.canhub.cropper.CropImageContract
import com.canhub.cropper.CropImageContractOptions
import com.canhub.cropper.CropImageOptions
import com.canhub.cropper.CropImageView
import com.example.detector.ml.Eyescorer
import com.example.detector.ml.SkinTriage
import com.example.detector.ui.theme.DetectorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import kotlin.system.measureTimeMillis

private const val IMAGE_SIZE = 224
private const val FILE_PROVIDER_AUTHORITY = "com.example.detector.provider"
private const val SKIN_REFERRAL_THRESHOLD = 0.715f
private const val EYE_REFERRAL_THRESHOLD = 0.515f

enum class AnalysisMode(val label: String) {
    SKIN("Skin"),
    EYE("Eye")
}

data class AnalysisResult(
    val mode: AnalysisMode,
    val score: Float,
    val elapsedMs: Long
)

class MainActivity : ComponentActivity() {
    private lateinit var tempImageUri: Uri

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        tempImageUri = createTempImageUri()

        setContent {
            DetectorTheme(darkTheme = true, dynamicColor = false) {
                val navController = rememberNavController()

                NavHost(navController = navController, startDestination = "detector") {
                    composable("detector") {
                        DetectorScreen(
                            tempImageUri = tempImageUri,
                            onCreateTempUri = {
                                createTempImageUri().also { tempImageUri = it }
                            },
                            onAnalysisComplete = { result ->
                                navController.navigate("result/${result.mode.name}/${result.score}/${result.elapsedMs}")
                            }
                        )
                    }
                    composable(
                        route = "result/{mode}/{score}/{elapsedMs}",
                        arguments = listOf(
                            navArgument("mode") { type = NavType.StringType },
                            navArgument("score") { type = NavType.FloatType },
                            navArgument("elapsedMs") { type = NavType.LongType }
                        )
                    ) { backStackEntry ->
                        val mode = backStackEntry.arguments
                            ?.getString("mode")
                            ?.let { runCatching { AnalysisMode.valueOf(it) }.getOrNull() }
                            ?: AnalysisMode.SKIN
                        val result = AnalysisResult(
                            mode = mode,
                            score = backStackEntry.arguments?.getFloat("score") ?: 0f,
                            elapsedMs = backStackEntry.arguments?.getLong("elapsedMs") ?: 0L
                        )

                        ResultScreen(result = result, onBack = navController::popBackStack)
                    }
                }
            }
        }
    }

    private fun createTempImageUri(): Uri {
        val file = File(cacheDir, "capture-${System.currentTimeMillis()}.jpg")
        return FileProvider.getUriForFile(this, FILE_PROVIDER_AUTHORITY, file)
    }
}

@Composable
fun DetectorScreen(
    tempImageUri: Uri,
    onCreateTempUri: () -> Uri,
    onAnalysisComplete: (AnalysisResult) -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var croppedImageUri by remember { mutableStateOf<Uri?>(null) }
    var captureUri by remember(tempImageUri) { mutableStateOf(tempImageUri) }
    var activeMode by remember { mutableStateOf<AnalysisMode?>(null) }
    val isAnalyzing = activeMode != null

    fun cropOptions(uri: Uri) = CropImageContractOptions(
        uri = uri,
        cropImageOptions = CropImageOptions(
            guidelines = CropImageView.Guidelines.ON,
            aspectRatioX = IMAGE_SIZE,
            aspectRatioY = IMAGE_SIZE,
            outputRequestWidth = IMAGE_SIZE,
            outputRequestHeight = IMAGE_SIZE,
            outputRequestSizeOptions = CropImageView.RequestSizeOptions.RESIZE_EXACT,
            fixAspectRatio = true,
            cropShape = CropImageView.CropShape.RECTANGLE
        )
    )

    val cropImageLauncher = rememberLauncherForActivityResult(CropImageContract()) { result ->
        if (result.isSuccessful) {
            croppedImageUri = result.uriContent
        } else {
            Toast.makeText(context, "Image crop failed. Please try again.", Toast.LENGTH_SHORT).show()
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            cropImageLauncher.launch(cropOptions(captureUri))
        } else {
            Toast.makeText(context, "No image was captured.", Toast.LENGTH_SHORT).show()
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            cameraLauncher.launch(captureUri)
        } else {
            Toast.makeText(context, "Camera permission is required to take a picture.", Toast.LENGTH_SHORT).show()
        }
    }

    fun startCapture() {
        captureUri = onCreateTempUri()
        when (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)) {
            PackageManager.PERMISSION_GRANTED -> cameraLauncher.launch(captureUri)
            else -> permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    fun analyze(mode: AnalysisMode) {
        val imageUri = croppedImageUri
        if (imageUri == null) {
            Toast.makeText(context, "Take and crop a picture first.", Toast.LENGTH_SHORT).show()
            return
        }

        activeMode = mode
        scope.launch {
            val result = withContext(Dispatchers.Default) {
                val bitmap = context.decodeBitmap(imageUri)
                if (bitmap == null) {
                    null
                } else {
                    var score = 0f
                    val elapsedMs = measureTimeMillis {
                        score = runModel(context, bitmap, mode)
                    }
                    AnalysisResult(mode = mode, score = score, elapsedMs = elapsedMs)
                }
            }

            activeMode = null
            if (result == null) {
                Toast.makeText(context, "Could not read the selected image.", Toast.LENGTH_SHORT).show()
            } else {
                onAnalysisComplete(result)
            }
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 24.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            AppHeader()
            ImagePreview(imageUri = croppedImageUri, context = context)
            ActionPanel(
                hasImage = croppedImageUri != null,
                isAnalyzing = isAnalyzing,
                activeMode = activeMode,
                onCapture = ::startCapture,
                onAnalyzeSkin = { analyze(AnalysisMode.SKIN) },
                onAnalyzeEye = { analyze(AnalysisMode.EYE) }
            )
        }
    }
}

@Composable
private fun AppHeader() {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.Start,
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = "Vital",
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = "Offline disease screening for low-resource clinics",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}

@Composable
private fun ImagePreview(imageUri: Uri?, context: Context) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(300.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.28f)),
        contentAlignment = Alignment.Center
    ) {
        val bitmap = imageUri?.let { uri ->
            remember(uri) { context.decodeBitmap(uri) }
        }

        if (bitmap != null) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Cropped image preview",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        } else {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(24.dp)
            ) {
                Text(
                    text = "No image selected",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = "Take a close, well-lit photo and crop the relevant region.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.72f),
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
private fun ActionPanel(
    hasImage: Boolean,
    isAnalyzing: Boolean,
    activeMode: AnalysisMode?,
    onCapture: () -> Unit,
    onAnalyzeSkin: () -> Unit,
    onAnalyzeEye: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Button(
            onClick = onCapture,
            enabled = !isAnalyzing,
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp),
            shape = RoundedCornerShape(6.dp)
        ) {
            Text(
                text = if (hasImage) "Retake Photo" else "Take Photo",
                fontWeight = FontWeight.SemiBold
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            AnalysisButton(
                label = "Skin",
                isLoading = activeMode == AnalysisMode.SKIN,
                enabled = hasImage && !isAnalyzing,
                onClick = onAnalyzeSkin,
                modifier = Modifier.weight(1f)
            )
            AnalysisButton(
                label = "Eye",
                isLoading = activeMode == AnalysisMode.EYE,
                enabled = hasImage && !isAnalyzing,
                onClick = onAnalyzeEye,
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
private fun AnalysisButton(
    label: String,
    isLoading: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled || isLoading,
        modifier = modifier.height(50.dp),
        shape = RoundedCornerShape(6.dp)
    ) {
        if (isLoading) {
            CircularProgressIndicator(
                modifier = Modifier
                    .height(18.dp)
                    .width(18.dp),
                strokeWidth = 2.dp
            )
            Spacer(modifier = Modifier.width(8.dp))
        }
        Text(if (isLoading) "Running" else "Analyze $label")
    }
}

@Composable
fun ResultScreen(result: AnalysisResult, onBack: () -> Unit) {
    val assessment = assessmentFor(result.score)

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "${result.mode.label} Triage",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(16.dp))
            Text(
                text = "%.2f".format(result.score),
                fontSize = 58.sp,
                color = Color.White,
                fontWeight = FontWeight.ExtraBold
            )
            Text(
                text = "local inference in ${result.elapsedMs} ms",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(18.dp))
            Text(
                text = assessment.title,
                style = MaterialTheme.typography.titleLarge,
                color = assessment.color,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(8.dp))
            Text(
                text = assessment.message,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(Modifier.height(34.dp))
            Button(
                onClick = onBack,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),
                shape = RoundedCornerShape(6.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Analyze Another Image", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

private data class Assessment(
    val title: String,
    val message: String,
    val color: Color
)

private fun assessmentFor(score: Float): Assessment = when {
    score >= 5f -> Assessment(
        title = "High-priority referral",
        message = "The local model produced an elevated signal. Treat this as a reason to seek qualified clinical review, not as a diagnosis.",
        color = Color(0xFFFFB4AB)
    )
    score >= 4f -> Assessment(
        title = "Repeat or review",
        message = "The model output is near the caution range. Retake the image in better lighting or route the case for clinical review.",
        color = Color(0xFFFFD166)
    )
    else -> Assessment(
        title = "Lower signal",
        message = "No significant abnormality was detected by this prototype. Continue monitoring symptoms and use clinical care for concerns.",
        color = Color(0xFF8EE3C8)
    )
}

fun runModel(context: Context, bitmap: Bitmap, mode: AnalysisMode): Float = try {
    val input = bitmap.toTensorImage().tensorBuffer
    val (probability, referralThreshold) = when (mode) {
        AnalysisMode.SKIN -> {
            val model = SkinTriage.newInstance(context)
            try {
                model.process(input).outputFeature0AsTensorBuffer.floatArray.firstOrNull() ?: 0f
            } finally {
                model.close()
            } to SKIN_REFERRAL_THRESHOLD
        }
        AnalysisMode.EYE -> {
            val model = Eyescorer.newInstance(context)
            try {
                model.process(input).outputFeature0AsTensorBuffer.floatArray.firstOrNull() ?: 0f
            } finally {
                model.close()
            } to EYE_REFERRAL_THRESHOLD
        }
    }
    referralScore(probability, referralThreshold)
} catch (e: Exception) {
    e.printStackTrace()
    0f
}

private fun referralScore(probability: Float, referralThreshold: Float): Float {
    return ((probability.coerceIn(0f, 1f) / referralThreshold) * 5f).coerceIn(0f, 10f)
}

private fun Bitmap.toTensorImage(): TensorImage {
    val processor = ImageProcessor.Builder()
        .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    val image = TensorImage(DataType.FLOAT32)
    image.load(this)
    return processor.process(image)
}

private fun Context.decodeBitmap(uri: Uri): Bitmap? {
    return contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
}
