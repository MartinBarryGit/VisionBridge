import javax.sound.sampled.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.collections.ArrayDeque
import kotlin.math.*

// Constants matching Python version
const val SAMPLE_RATE = 16000
const val CHUNK_SIZE = 512
const val FRAME_MS = (CHUNK_SIZE.toDouble() / SAMPLE_RATE) * 1000.0  // ~32 ms
const val PRE_ROLL_MS = 100
const val NUM_PRE_ROLL_FRAMES = (PRE_ROLL_MS / FRAME_MS).toInt()  // ~3 frames

class SpeechDetector {
    private var recording = false
    private var properStartSent = false
    private val frames = mutableListOf<ByteArray>()
    
    // Buffers for speech segment detection
    private val audioBuffer = ByteArrayOutputStream()
    private var triggered = false
    
    // Ring buffer for pre-roll
    private val ringBuffer = ArrayDeque<ByteArray>(NUM_PRE_ROLL_FRAMES)
    
    // VAD buffer
    private val vadBuffer = mutableListOf<Float>()
    
    // Simple VAD implementation (placeholder for Silero VAD)
    private var silenceCounter = 0
    private var speechCounter = 0
    
    init {
        println("Speech Detector initialized")
        println("Sample rate: $SAMPLE_RATE Hz, Chunk size: $CHUNK_SIZE samples (~${FRAME_MS.toInt()} ms)")
    }
    
    fun recordAudio(savePath: String = "recorded_audio.wav"): Int {
        println("Listening for speech...")
        
        val audioFormat = AudioFormat(
            AudioFormat.Encoding.PCM_SIGNED,
            SAMPLE_RATE.toFloat(),
            16, // 16-bit
            1,  // mono
            2,  // frame size: 1 channel * 2 bytes per sample
            SAMPLE_RATE.toFloat(),
            false // little endian
        )
        
        val info = DataLine.Info(TargetDataLine::class.java, audioFormat)
        
        if (!AudioSystem.isLineSupported(info)) {
            println("Audio recording not supported!")
            return -1
        }
        
        val line = AudioSystem.getLine(info) as TargetDataLine
        
        try {
            line.open(audioFormat)
            line.start()
            println("🎤 Recording started...")
            
            val buffer = ByteArray(CHUNK_SIZE * 2) // 2 bytes per sample
            
            while (true) {
                val bytesRead = line.read(buffer, 0, buffer.size)
                
                if (bytesRead > 0) {
                    val end_recording = processAudioChunk(buffer, bytesRead, savePath)
                    // Check if we should stop recording
                    if (end_recording) {
                        // Recording complete
                        break
                    }
                }
                
                // Check for user interrupt (simple timeout for demo)
                if (silenceCounter > 1000) { // ~32 seconds of silence
                    println("Recording stopped due to long silence")
                    break
                }
            }
            
        } catch (e: Exception) {
            println("Recording error: ${e.message}")
            return -1
        } finally {
            line.stop()
            line.close()
        }
        
        return 0
    }
    
    private fun processAudioChunk(buffer: ByteArray, bytesRead: Int, savePath: String): Boolean {
        // Convert bytes to float array for VAD processing
        val samples = bytesRead / 2 // 2 bytes per sample
        val audioFloat = FloatArray(samples)
        
        val byteBuffer = ByteBuffer.wrap(buffer, 0, bytesRead).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until samples) {
            val sample = byteBuffer.short
            audioFloat[i] = sample.toFloat() / 32768.0f
        }
        
        vadBuffer.addAll(audioFloat.toList())
        
        while (vadBuffer.size >= CHUNK_SIZE) {
            val currentChunk = vadBuffer.take(CHUNK_SIZE).toFloatArray()
            repeat(CHUNK_SIZE) { vadBuffer.removeFirst() }
            
            // Simple VAD (replace with actual Silero VAD if needed)
            val (isSpeechStart, isSpeechEnd) = simpleVAD(currentChunk)
            
            val chunkBytes = floatArrayToBytes(currentChunk)
            
            if (isSpeechStart && !triggered) {
                println("🎤 Speech START detected")
                
                // Prepend pre-roll frames
                for (rbChunk in ringBuffer) {
                    audioBuffer.write(rbChunk)
                }
                ringBuffer.clear()
                triggered = true
                properStartSent = false
            }
            
            if (triggered) {
                audioBuffer.write(chunkBytes)
                
                // Check if buffer meets threshold and proper start not sent
                if (audioBuffer.size() >= 24000 && !properStartSent) {
                    println("✅ Proper speech start detected (>=0.75s)")
                    properStartSent = true
                }
            } else {
                // Store in ring buffer for pre-roll
                if (ringBuffer.size >= NUM_PRE_ROLL_FRAMES) {
                    ringBuffer.removeFirst()
                }
                ringBuffer.addLast(chunkBytes)
            }
            
            if (isSpeechEnd && triggered) {
                println("🛑 Speech END detected")
                triggered = false
                
                if (audioBuffer.size() >= 24000) {
                    // Save the recorded audio
                    saveAudioToFile(audioBuffer.toByteArray(), savePath)
                    val duration = audioBuffer.size().toDouble() / (SAMPLE_RATE * 2)
                    println("💾 Audio saved: $duration")

                    // Clear buffer for next utterance
                    audioBuffer.reset()
                    properStartSent = false
                    return true
                }
            }
        
        }
        return false
    }
    
    private fun simpleVAD(audioChunk: FloatArray): Pair<Boolean, Boolean> {
        // Simple energy-based VAD (replace with Silero VAD for production)
        val energy = audioChunk.map { it * it }.average()
        val threshold = 0.001 // Adjust based on your environment
        
        val isSpeech = energy > threshold
        
        var speechStart = false
        var speechEnd = false
        
        if (isSpeech) {
            speechCounter++
            silenceCounter = 0
            
            // Start speech if we have enough consecutive speech frames
            if (speechCounter == 3 && !triggered) {
                speechStart = true
            }
        } else {
            silenceCounter++
            speechCounter = 0
            
            // End speech if we have enough consecutive silence frames
            if (silenceCounter == 10 && triggered) {
                speechEnd = true
            }
        }
        
        return Pair(speechStart, speechEnd)
    }
    
    private fun floatArrayToBytes(floatArray: FloatArray): ByteArray {
        val byteArray = ByteArray(floatArray.size * 2)
        val byteBuffer = ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN)
        
        for (sample in floatArray) {
            val intSample = (sample * 32767).toInt().coerceIn(-32768, 32767)
            byteBuffer.putShort(intSample.toShort())
        }
        
        return byteArray
    }
    
    private fun saveAudioToFile(audioData: ByteArray, filename: String) {
        try {
            val audioFormat = AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                SAMPLE_RATE.toFloat(),
                16, // 16-bit
                1,  // mono
                2,  // frame size
                SAMPLE_RATE.toFloat(),
                false // little endian
            )
            
            val audioInputStream = AudioInputStream(
                ByteArrayInputStream(audioData),
                audioFormat,
                audioData.size / audioFormat.frameSize.toLong()
            )
            
            val file = File(filename)
            AudioSystem.write(audioInputStream, AudioFileFormat.Type.WAVE, file)
            println("Audio saved to: ${file.absolutePath}")
            
        } catch (e: Exception) {
            println("Error saving audio: ${e.message}")
        }
    }
}

fun main() {
    println("=== Speech Detector Test ===")
    
    val detector = SpeechDetector()
    
    println("Starting speech detection...")
    println("Speak into your microphone. The detector will:")
    println("- Start recording when speech is detected")
    println("- Stop recording after speech ends")
    println("- Save audio to 'recorded_audio.wav'")
    println("\nPress Ctrl+C to stop")
    
    try {
        val result = detector.recordAudio("test_recording.wav")
        
        if (result == 0) {
            println("Recording completed successfully!")
        } else {
            println("Recording failed with code: $result")
        }
        
    } catch (e: Exception) {
        println("Error during recording: ${e.message}")
    }
    
    println("Speech detection test complete!")
}
