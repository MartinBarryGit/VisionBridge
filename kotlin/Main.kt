import kotlinx.coroutines.*
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Paths
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.IIOImage
import javax.imageio.ImageWriteParam
import kotlin.random.Random

import ai.onnxruntime.*
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.nio.FloatBuffer
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import org.opencv.highgui.HighGui

class VisionBridgeMain {
    private val soundPlayer = SoundPlayer()
    private val speechDetector = SpeechDetector()
    private val aiDescriptor = AIDoorDescriptor()
    private val client = HttpClient.newBuilder().build()
    private val apiKey = System.getenv("OPENAI_API_KEY")
    
    suspend fun main() {
        println("=== VisionBridge Kotlin Main ===")
        println("Initializing components...")
        
        if (apiKey.isNullOrEmpty()) {
            println("❌ OPENAI_API_KEY not set!")
            return
        }
        
        // Initialize camera (simulated for now)
        println("✅ Camera initialized (simulated)")
        println("✅ Sound player ready")
        println("✅ Speech detector ready")
        println("✅ AI descriptor ready")
        
        println("\nStarting main loop...")
        println("Say 'OK Antoine' followed by your command...")
        
        while (true) {
            try {
                // Step 1: Record audio
                println("\n🎤 Listening for speech...")
                val recordResult = speechDetector.recordAudio("recorded_audio.wav")
                
                if (recordResult != 0) {
                    println("Recording failed, retrying...")
                    delay(1000)
                    continue
                }
                println("✅ Audio recorded successfully")
                // Step 2: Transcribe audio using OpenAI Whisper
                val transcription = transcribeAudio("recorded_audio.wav")
                
                if (transcription.isEmpty()) {
                    println("No transcription received, continuing...")
                    continue
                }
                
                println("🗣️ Transcribed: \"$transcription\"")
                
                // Step 3: Check for wake word
                val wakeWord = "antoine"
                val normalizedTranscription = transcription.lowercase().trim()
                
                if (!normalizedTranscription.contains(wakeWord)) {
                    println("Wake word not detected, continuing to listen...")
                    continue
                }
                
                println("✅ Wake word 'OK Antoine' detected!")
                
                // Step 4: Extract user command
                val userInput = normalizedTranscription.replace(wakeWord, "").trim()
                println("📝 User command: \"$userInput\"")
                
                // Step 5: Capture frame (simulated)
                liveDetection()
                
                
               
                // // Step 7: Generate TTS response
                // println("🔊 Generating speech response...")
                // speakText(frameDescription)
                
                // // Step 8: Check if we should start door detection
                // if (shouldStartDoorDetection(frameDescription)) {
                //     println("🚪 Starting door detection mode...")
                //     runDoorDetectionLoop()
                // }
                
            } catch (e: Exception) {
                println("❌ Error in main loop: ${e.message}")
                delay(2000)
            }
        }
    }
    
    private suspend fun transcribeAudio(audioPath: String): String {
        return try {
            println("🎵 Transcribing audio with OpenAI Whisper...")
            
            val audioFile = File(audioPath)
            if (!audioFile.exists()) {
                println("Audio file not found: $audioPath")
                return ""
            }
            
            // Read audio file as bytes
            val audioBytes = audioFile.readBytes()
            
            // Create multipart form data
            val boundary = "----VisionBridgeBoundary${System.currentTimeMillis()}"
            val requestBody = buildMultipartBody(boundary, audioBytes, "audio.wav")
            
            // Create HTTP request
            val request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.openai.com/v1/audio/transcriptions"))
                .header("Authorization", "Bearer $apiKey")
                .header("Content-Type", "multipart/form-data; boundary=$boundary")
                .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody))
                .build()
            
            // Send request
            val response = client.send(request, HttpResponse.BodyHandlers.ofString())
            
            if (response.statusCode() == 200) {
                // Parse JSON response to extract transcription
                val transcription = parseTranscriptionResponse(response.body())
                println("✅ Transcription successful: \"$transcription\"")
                return transcription
            } else {
                println("❌ Transcription failed: ${response.statusCode()} - ${response.body()}")
                return ""
            }
            
        } catch (e: Exception) {
            println("Error transcribing audio: ${e.message}")
            e.printStackTrace()
            ""
        }
    }
    private fun buildMultipartBody(boundary: String, audioBytes: ByteArray, filename: String): ByteArray {
        val output = ByteArrayOutputStream()
        val writer = output.writer(Charsets.UTF_8)
        
        // File field
        writer.write("--$boundary\r\n")
        writer.write("Content-Disposition: form-data; name=\"file\"; filename=\"$filename\"\r\n")
        writer.write("Content-Type: audio/wav\r\n")
        writer.write("\r\n")
        writer.flush()
        output.write(audioBytes)
        writer.write("\r\n")
        
        // Model field
        writer.write("--$boundary\r\n")
        writer.write("Content-Disposition: form-data; name=\"model\"\r\n")
        writer.write("\r\n")
        writer.write("whisper-1")
        writer.write("\r\n")
        
        // Language field (optional)
        writer.write("--$boundary\r\n")
        writer.write("Content-Disposition: form-data; name=\"language\"\r\n")
        writer.write("\r\n")
        writer.write("fr")
        writer.write("\r\n")
        
        // Response format field
        writer.write("--$boundary\r\n")
        writer.write("Content-Disposition: form-data; name=\"response_format\"\r\n")
        writer.write("\r\n")
        writer.write("json")
        writer.write("\r\n")
        
        // End boundary
        writer.write("--$boundary--\r\n")
        writer.flush()
        
        return output.toByteArray()
    }
    
    private fun parseTranscriptionResponse(responseBody: String): String {
        return try {
            // Simple JSON parsing to extract "text" field
            val textPattern = "\"text\"\\s*:\\s*\"([^\"]+)\"".toRegex()
            val match = textPattern.find(responseBody)
            
            if (match != null) {
                match.groupValues[1]
                    .replace("\\n", " ")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                    .trim()
            } else {
                println("Could not parse transcription from response: $responseBody")
                ""
            }
        } catch (e: Exception) {
            println("Error parsing transcription response: ${e.message}")
            ""
        }
    }
    suspend fun liveDetection() {
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
    var door_status = -1
    val frame = Mat()
    while (true) {
        val ret = cap.read(frame)
        if (!ret || frame.empty()) {
            println("Impossible de lire une frame")
            break
        }
        val bufferedImage = matToBufferedImage(frame)
        // Convert Mat to BufferedImage
        if (door_status != 0) {
            
            val structuredResponse = describeFrameWithAI(bufferedImage)
            val answer = structuredResponse.answer
            door_status = structuredResponse.status
            if (door_status == 0) {
                println("🎯 AI Description: $answer")
            } else if (door_status == -1) {
                return
            }
        }
        // Run detection
        val yoloResponse = boxSoundDetection(bufferedImage, model)
        val annotated = yoloResponse.annotated
        val doorPosition = yoloResponse.position
        
        // Convert back to Mat and show
        val annotatedMat = bufferedImageToMat(annotated)
        HighGui.imshow("YOLOv8 Live Detection (EMA + IoU + Tracking)", annotatedMat)
        val (stereoSound, rate) = computeDirectionalSound(doorPosition, userHeadingDeg = 0.0, targetPos = Array<Float>(2) { 0f })
        soundPlayer.play(stereoSound, rate)

        // Check for 'q' key
        if (HighGui.waitKey(1) == 'q'.code) {
            break
        }
    }
    
    cap.release()
    HighGui.destroyAllWindows()
    model.close()
}
   
    private suspend fun describeFrameWithAI(frame: BufferedImage): DoorDetectionResponse {
        try {
            println("🖼️ Processing image with AI descriptor...")
            
            // Use our AI descriptor with the actual frame
            val description = aiDescriptor.describeDoor(frame)
            
            println("🎯 AI Description: $description")
            return description
            
        } catch (e: Exception) {
            println("Error describing frame: ${e.message}")
            return DoorDetectionResponse(
                doors = emptyList(),
                answer = "I can see the scene but encountered an error processing it.",
                status = -1
            )
        }
    }
    
    private suspend fun speakText(text: String) {
        try {
            println("🗣️ Speaking: \"$text\"")
            
            // For now, just print the text
            // In a real implementation, you would:
            // 1. Call OpenAI TTS API to generate audio
            // 2. Play the audio using your SoundPlayer
            
            println("⚠️ TTS not implemented - would speak: \"$text\"")
            
            // Simulate speaking duration
            delay(text.length * 50L) // ~50ms per character
            
        } catch (e: Exception) {
            println("Error speaking text: ${e.message}")
        }
    }
    
}

suspend fun main() {
    val app = VisionBridgeMain()
    app.main()
}

// // Extension function to run the main function
// fun main(args: Array<String>) {
//     runBlocking {
//         main()
//     }
// }
