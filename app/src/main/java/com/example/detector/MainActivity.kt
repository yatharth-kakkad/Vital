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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
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
import com.example.detector.ml.Dataprep
import com.example.detector.ml.Eyescorer
import com.example.detector.ml.Predictor
import com.example.detector.ml.ScorerMob
import com.example.detector.ml.ScorerRes
import com.example.detector.ui.theme.DetectorTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import kotlin.math.min

private const val IMAGE_SIZE = 224
private const val FILE_PROVIDER_AUTHORITY = "com.example.detector.provider"

enum class AnalysisMode(val label: String) {
    SKIN("Skin"),
    EYE("Eye")
}

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
                            onAnalysisComplete = { mode, score ->
                                navController.navigate("result/${mode.name}/$score")
                            }
                        )
                    }
                    composable(
                        route = "result/{mode}/{score}",
                        arguments = listOf(
                            navArgument("mode") { type = NavType.StringType },
                            navArgument("score") { type = NavType.FloatType }
                        )
                    ) { backStackEntry ->
                        val mode = backStackEntry.arguments
                            ?.getString("mode")
                            ?.let { runCatching { AnalysisMode.valueOf(it) }.getOrNull() }
                            ?: AnalysisMode.SKIN
                        val score = backStackEntry.arguments?.getFloat("score") ?: 0f

                        ResultScreen(mode = mode, score = score, onBack = navController::popBackStack)
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
    onAnalysisComplete: (AnalysisMode, Float) -> Unit
) {
    val context = LocalContext.current
    var croppedImageUri by remember { mutableStateOf<Uri?>(null) }
    var captureUri by remember(tempImageUri) { mutableStateOf(tempImageUri) }

    fun launchCrop(uri: Uri, launcher: androidx.activity.compose.ManagedActivityResultLauncher<CropImageContractOptions, CropImageView.CropResult>) {
        launcher.launch(
            CropImageContractOptions(
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
        )
    }

    val cropImageLauncher = rememberLauncherForActivityResult(CropImageContract()) { result ->
        if (result.isSuccessful) {
            croppedImageUri = result.uriContent
        } else {
            Toast.makeText(context, "Image crop failed. Please try again.", Toast.LENGTH_SHORT).show()
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            launchCrop(captureUri, cropImageLauncher)
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

        val bitmap = context.decodeBitmap(imageUri)
        if (bitmap == null) {
            Toast.makeText(context, "Could not read the selected image.", Toast.LENGTH_SHORT).show()
            return
        }

        val score = when (mode) {
            AnalysisMode.SKIN -> runSkinModel(context, bitmap)
            AnalysisMode.EYE -> runEyeModel(context, bitmap)
        }
        onAnalysisComplete(mode, score)
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 28.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            AppHeader()
            ImagePreview(imageUri = croppedImageUri, context = context)
            ActionPanel(
                hasImage = croppedImageUri != null,
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
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Vital",
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.ExtraBold,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = "Camera-based skin and eye screening prototype",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground
        )
        Text(
            text = "Capture a clear, well-lit close-up, crop it to the area of concern, then run one of the on-device TensorFlow Lite models.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun ImagePreview(imageUri: Uri?, context: Context) {
    val previewShape = RoundedCornerShape(28.dp)

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(340.dp)
            .clip(previewShape)
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
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "No image selected",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = "Use the camera to add a 224 x 224 crop.",
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
    onCapture: () -> Unit,
    onAnalyzeSkin: () -> Unit,
    onAnalyzeEye: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Button(
                onClick = onCapture,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                shape = RoundedCornerShape(14.dp)
            ) {
                Text(
                    text = if (hasImage) "Retake Photo" else "Take Photo",
                    fontWeight = FontWeight.SemiBold
                )
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(
                    onClick = onAnalyzeSkin,
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Text("Analyze Skin")
                }
                OutlinedButton(
                    onClick = onAnalyzeEye,
                    modifier = Modifier
                        .weight(1f)
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Text("Analyze Eye")
                }
            }
        }
    }
}

@Composable
fun ResultScreen(mode: AnalysisMode, score: Float, onBack: () -> Unit) {
    val assessment = assessmentFor(score)

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "${mode.label} Analysis",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(18.dp))
            Text(
                text = "%.2f".format(score),
                fontSize = 64.sp,
                color = Color.White,
                fontWeight = FontWeight.ExtraBold
            )
            Spacer(Modifier.height(16.dp))
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
            Spacer(Modifier.height(36.dp))
            Button(
                onClick = onBack,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
                shape = RoundedCornerShape(14.dp),
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
        title = "Elevated signal detected",
        message = "This prototype found a stronger abnormality signal. Please consult a qualified clinician for a real diagnosis.",
        color = Color(0xFFFFB4AB)
    )
    score >= 4f -> Assessment(
        title = "Borderline signal",
        message = "The model output is near the caution range. Retake the photo in better lighting or seek professional guidance.",
        color = Color(0xFFFFD166)
    )
    else -> Assessment(
        title = "Lower signal",
        message = "No significant abnormality was detected by this prototype. Keep monitoring changes and use clinical care for concerns.",
        color = Color(0xFF8EE3C8)
    )
}

fun runSkinModel(context: Context, bitmap: Bitmap): Float {
    return try {
        val normalizedProcessor = ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0.0f, 255.0f))
            .build()

        val scoringProcessor = ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        var normalizedImage = TensorImage(DataType.FLOAT32)
        normalizedImage.load(bitmap)
        normalizedImage = normalizedProcessor.process(normalizedImage)

        var scoringImage = TensorImage(DataType.FLOAT32)
        scoringImage.load(bitmap)
        scoringImage = scoringProcessor.process(scoringImage)

        val dataprep = Dataprep.newInstance(context)
        val embedding = dataprep.process(normalizedImage.tensorBuffer).outputFeature0AsTensorBuffer
        dataprep.close()

        val predictorInput = TensorBuffer.createFixedSize(intArrayOf(1, 512), DataType.FLOAT32)
        predictorInput.loadArray(embedding.floatArray)

        val predictor = Predictor.newInstance(context)
        val predictorOutput = predictor.process(predictorInput).outputFeature0AsTensorBuffer.floatArray
        predictor.close()

        val scorerMob = ScorerMob.newInstance(context)
        val mobScore = scorerMob.process(scoringImage.tensorBuffer).outputFeature0AsTensorBuffer.floatArray
        scorerMob.close()

        val scorerRes = ScorerRes.newInstance(context)
        val resScore = scorerRes.process(scoringImage.tensorBuffer).outputFeature0AsTensorBuffer.floatArray
        scorerRes.close()

        val scoringTable = floatArrayOf(8f, 3f, 1f, 0f, 22f, 1f, 1f)
        val weightedMob = weightedSum(mobScore, scoringTable)
        val weightedRes = weightedSum(resScore, scoringTable)
        val predictorConfidence = predictorOutput.firstOrNull() ?: 0f

        if (predictorConfidence > 0f) weightedRes else (weightedMob + weightedRes) / 2f
    } catch (e: Exception) {
        e.printStackTrace()
        0f
    }
}

fun runEyeModel(context: Context, bitmap: Bitmap): Float {
    return try {
        val eyeImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        var eyeTensorImage = TensorImage(DataType.FLOAT32)
        eyeTensorImage.load(bitmap)
        eyeTensorImage = eyeImageProcessor.process(eyeTensorImage)

        val eyeScorer = Eyescorer.newInstance(context)
        val eyeScores = eyeScorer.process(eyeTensorImage.tensorBuffer).outputFeature0AsTensorBuffer.floatArray
        eyeScorer.close()

        weightedSum(eyeScores, floatArrayOf(5f, 10f, 1f, 8f))
    } catch (e: Exception) {
        e.printStackTrace()
        0f
    }
}

private fun weightedSum(scores: FloatArray, weights: FloatArray): Float {
    var total = 0f
    for (index in 0 until min(scores.size, weights.size)) {
        total += scores[index] * weights[index]
    }
    return total
}

private fun Context.decodeBitmap(uri: Uri): Bitmap? {
    return contentResolver.openInputStream(uri)?.use(BitmapFactory::decodeStream)
}
