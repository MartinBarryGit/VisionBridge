import ai.onnxruntime.*
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.image.BufferedImage
import java.io.File
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import kotlin.math.max
import kotlin.math.min
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import org.opencv.highgui.HighGui

// Global configuration
const val ALPHA = 0.2f
const val IOU_THRESHOLD = 0.5f
const val MAX_MISSED_FRAMES = 30
const val INPUT_SIZE = 640

class Track(var box: FloatArray, var score: Float) {
    var missed = 0
    
    fun update(newBox: FloatArray, newScore: Float) {
        // EMA smoothing like Python version
        for (i in box.indices) {
            box[i] = ALPHA * newBox[i] + (1 - ALPHA) * box[i]
        }
        score = ALPHA * newScore + (1 - ALPHA) * score
        missed = 0
    }
    
    fun markMissed() {
        missed += 1
    }
    
    fun isActive(): Boolean {
        return missed < MAX_MISSED_FRAMES
    }
}

fun iou(box1: FloatArray, box2: FloatArray): Float {
    val (x1, y1, x2, y2) = box1
    val (x1b, y1b, x2b, y2b) = box2

    val inter_x1 = max(x1, x1b)
    val inter_y1 = max(y1, y1b)
    val inter_x2 = min(x2, x2b)
    val inter_y2 = min(y2, y2b)

    val inter_area = max(0.0f, inter_x2 - inter_x1) * max(0.0f, inter_y2 - inter_y1)
    val area1 = (x2 - x1) * (y2 - y1)
    val area2 = (x2b - x1b) * (y2b - y1b)

    val union = area1 + area2 - inter_area
    return if (union > 0.0f) inter_area / union else 0.0f
}

// Global tracks list (like Python version)
var tracks = mutableListOf<Track>()

fun smoothWithTracking(newBoxes: List<FloatArray>, confThreshold: Float = 0.7f): Pair<List<FloatArray>, List<Float>> {
    val updatedTracks = MutableList(tracks.size) { false }
    val usedTrackIndices = mutableSetOf<Int>()
    for (box in newBoxes) {
        val score = box[4]
        
        if (score < confThreshold) continue
        var matched = false
        for (i in tracks.indices) {
            // if (i in usedTrackIndices) continue
            if (iou(box, tracks[i].box) > IOU_THRESHOLD) {
                tracks[i].update(box, score)
                updatedTracks[i] = true
                usedTrackIndices.add(i)
                matched = true
                break
            }
        }
        if (!matched) {
            tracks.add(Track(box.copyOf(), score))
            updatedTracks.add(true)
        }
    }

    for (i in updatedTracks.indices) {
        if (!updatedTracks[i]) {
            tracks[i].markMissed()
        }
    }

    tracks = tracks.filter { it.isActive() }.toMutableList()

    return Pair(tracks.map { it.box }, tracks.map { it.score })
}
// Simple ONNX YOLO detector (like Python YOLO model)
class YoloModel(modelPath: String) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    
    init {
        val sessionOptions = OrtSession.SessionOptions()
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelPath, sessionOptions)
        println("Model loaded: $modelPath")
    }
    
    fun predict(image: BufferedImage): List<FloatArray> {
        // Resize to 640x640
        val resized = BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB)
        val g = resized.createGraphics()
        g.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE, null)
        g.dispose()
        
        // Convert to CHW format [1, 3, 640, 640]
        val buffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getRGB(0, 0, INPUT_SIZE, INPUT_SIZE, pixels, 0, INPUT_SIZE)
        
        // R channel
        for (pixel in pixels) buffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        // G channel  
        for (pixel in pixels) buffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        // B channel
        for (pixel in pixels) buffer.put((pixel and 0xFF) / 255.0f)
        
        buffer.rewind()
        
        // Run inference
        val inputTensor = OnnxTensor.createTensor(env, buffer, longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
        val outputs = session.run(mapOf("images" to inputTensor))
        
        // Parse output [1, 5, 8400]
        val outputTensor = outputs[0].value as Array<*>
        @Suppress("UNCHECKED_CAST")
        val output = outputTensor[0] as Array<FloatArray>
        
        outputs.close()
        inputTensor.close()
        
        // Convert to boxes [x1, y1, x2, y2, conf]
        val scaleX = image.width.toFloat() / INPUT_SIZE
        val scaleY = image.height.toFloat() / INPUT_SIZE
        val boxes = mutableListOf<FloatArray>()
        
        for (i in 0 until output[0].size) {
            val conf = output[4][i]
            if (conf < 0.25f) continue
            
            val cx = output[0][i] * scaleX
            val cy = output[1][i] * scaleY
            val w = output[2][i] * scaleX
            val h = output[3][i] * scaleY
            
            val x1 = cx - w / 2
            val y1 = cy - h / 2
            val x2 = cx + w / 2
            val y2 = cy + h / 2
            
            boxes.add(floatArrayOf(x1, y1, x2, y2, conf))
        }
        
        return boxes
    }
    
    fun close() {
        session.close()
    }
}

// Main detection loop (like Python box_sound_detection)
fun boxSoundDetection(frame: BufferedImage, model: YoloModel): BufferedImage {
    val boxes = model.predict(frame)
    val (smoothedBoxes, smoothedScores) = smoothWithTracking(boxes)
    
    // Draw boxes
    val annotated = BufferedImage(frame.width, frame.height, BufferedImage.TYPE_INT_RGB)
    val g = annotated.createGraphics()
    g.drawImage(frame, 0, 0, null)
    g.stroke = BasicStroke(2f)
    g.font = Font("Arial", Font.BOLD, 16)
    
    smoothedBoxes.forEachIndexed { doorN, box ->
        val (x1, y1, x2, y2) = box
        val score = smoothedScores[doorN]
        
        g.color = Color.GREEN
        g.drawRect(x1.toInt(), y1.toInt(), (x2 - x1).toInt(), (y2 - y1).toInt())
        g.drawString("door %.2f".format(score), x1.toInt(), y1.toInt() - 10)
        
        if (doorN == 0) {
            val boxCenter = ((x1 + x2) / 2).toInt()
            val imageCenter = frame.width / 2
            // TODO: compute_directional_sound(imageCenter, 0, boxCenter)
        }
    }
    
    g.dispose()
    return annotated
}

// Live detection (like Python live_detection)
fun liveDetection() {
    // System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    nu.pattern.OpenCV.loadShared()
    val modelPath = "assets/best.onnx"
    if (!File(modelPath).exists()) {
        println("Model not found: $modelPath")
        return
    }
    
    val model = YoloModel(modelPath)
    val cap = VideoCapture(0)
    
    if (!cap.isOpened) {
        println("Impossible d'ouvrir la webcam")
        return
    }
    
    println("Appuie 'q' pour quitter.")
    
    val frame = Mat()
    while (true) {
        val ret = cap.read(frame)
        if (!ret || frame.empty()) {
            println("Impossible de lire une frame")
            break
        }
        
        // Convert Mat to BufferedImage
        val bufferedImage = matToBufferedImage(frame)
        
        // Run detection
        val annotated = boxSoundDetection(bufferedImage, model)
        
        // Convert back to Mat and show
        val annotatedMat = bufferedImageToMat(annotated)
        HighGui.imshow("YOLOv8 Live Detection (EMA + IoU + Tracking)", annotatedMat)
        
        // Check for 'q' key
        if (HighGui.waitKey(1) == 'q'.code) {
            break
        }
    }
    
    cap.release()
    HighGui.destroyAllWindows()
    model.close()
}

// Convert OpenCV Mat to BufferedImage
fun matToBufferedImage(mat: Mat): BufferedImage {
    val type = if (mat.channels() == 1) BufferedImage.TYPE_BYTE_GRAY else BufferedImage.TYPE_3BYTE_BGR
    val bufferSize = mat.channels() * mat.cols() * mat.rows()
    val buffer = ByteArray(bufferSize)
    mat.get(0, 0, buffer)
    
    val image = BufferedImage(mat.cols(), mat.rows(), type)
    val targetPixels = (image.raster.dataBuffer as java.awt.image.DataBufferByte).data
    System.arraycopy(buffer, 0, targetPixels, 0, buffer.size)
    
    // Convert BGR to RGB
    if (mat.channels() == 3) {
        for (i in 0 until bufferSize step 3) {
            val temp = targetPixels[i]
            targetPixels[i] = targetPixels[i + 2]
            targetPixels[i + 2] = temp
        }
    }
    
    return image
}

// Convert BufferedImage to OpenCV Mat
fun bufferedImageToMat(image: BufferedImage): Mat {
    val mat = Mat(image.height, image.width, CvType.CV_8UC3)
    val pixels = ByteArray(image.width * image.height * 3)
    
    for (y in 0 until image.height) {
        for (x in 0 until image.width) {
            val rgb = image.getRGB(x, y)
            val idx = (y * image.width + x) * 3
            pixels[idx] = ((rgb shr 16) and 0xFF).toByte()      // R -> B (OpenCV wants BGR)
            pixels[idx + 1] = ((rgb shr 8) and 0xFF).toByte()   // G -> G
            pixels[idx + 2] = (rgb and 0xFF).toByte()           // B -> R
        }
    }
    
    mat.put(0, 0, pixels)
    return mat
}

fun main() {
    liveDetection()
}