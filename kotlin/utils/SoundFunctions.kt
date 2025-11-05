import javax.sound.sampled.*
import kotlin.math.*

const val SR = 48000

class SoundPlayer {
    private var lastSound = System.currentTimeMillis() - 10000
    var rate = 0.0
    
    fun play(stereo: Array<FloatArray>, rate: Double) {
        val diff = (System.currentTimeMillis() - lastSound)
        if (diff < rate * 1000) {
            return
        }
        print("Playing sound...")
        try {
            playAudio(stereo)
            lastSound = System.currentTimeMillis()
        } catch (e: Exception) {
            println("Error playing audio: ${e.message}")
        }
    }
    
    private fun playAudio(stereo: Array<FloatArray>) {
        val audioFormat = AudioFormat(
            AudioFormat.Encoding.PCM_SIGNED,
            SR.toFloat(),
            16, // 16-bit
            2,  // stereo
            4,  // frame size: 2 channels * 2 bytes per sample
            SR.toFloat(),
            false // little endian
        )
        
        val info = DataLine.Info(SourceDataLine::class.java, audioFormat)
        
        if (!AudioSystem.isLineSupported(info)) {
            println("Audio line not supported!")
            return
        }
        
        val line = AudioSystem.getLine(info) as SourceDataLine
        
        line.open(audioFormat, SR * 4) // Buffer size
        line.start()
        print("✓ Audio line opened. ")
        
        // Convert float array to byte array
        val byteArray = convertToByteArray(stereo)
        println("Writing ${byteArray.size} bytes to audio line...")
        
        val bytesWritten = line.write(byteArray, 0, byteArray.size)
        println("Wrote $bytesWritten bytes. ")
        
        line.drain() // Wait for all data to be played
        line.stop()
        line.close()
        println("Audio playback complete.")
    }
    
    private fun convertToByteArray(stereo: Array<FloatArray>): ByteArray {
        val samples = stereo.size
        val byteArray = ByteArray(samples * 4) // 2 bytes per sample, 2 channels
        
        for (i in 0 until samples) {
            // Clamp and convert to 16-bit integers
            val leftSample = (stereo[i][0] * 32767).toInt().coerceIn(-32768, 32767).toShort()
            val rightSample = (stereo[i][1] * 32767).toInt().coerceIn(-32768, 32767).toShort()
            
            // Little endian format: low byte first, then high byte
            byteArray[i * 4] = (leftSample.toInt() and 0xFF).toByte()
            byteArray[i * 4 + 1] = ((leftSample.toInt() shr 8) and 0xFF).toByte()
            byteArray[i * 4 + 2] = (rightSample.toInt() and 0xFF).toByte()
            byteArray[i * 4 + 3] = ((rightSample.toInt() shr 8) and 0xFF).toByte()
        }
        
        println("Converted ${samples} samples to ${byteArray.size} bytes")
        return byteArray
    }
}

fun computeDirectionalSound(
    userPos: Array<Float>,
    userHeadingDeg: Double,
    targetPos: Array<Float>
): Pair<Array<FloatArray>, Double> {
    val dist = hypot(targetPos[0] - userPos[0], targetPos[1] - userPos[1])
    val relAngle = angleToRelative(userPos, userHeadingDeg, targetPos)
    val rate = 2.0 // distance_to_rate(dist)
    val sign = if (targetPos[0] - userPos[0] > 0) 1 else -1
    var interval = 1.0 / rate
    
    var stereo_amp = if (sign > 0) floatArrayOf(1.0f, 0.0f) else floatArrayOf(0.0f, 1.0f)
    
    if (abs(dist) < 1) {
        stereo_amp = floatArrayOf(0.5f, 0.5f)
        println("Target is straight ahead!")
        interval *= 0.5
    }
    val tone = makeTone(freq = 440.0, duration = .5, amp = 0.8f)
    val stereo = Array(tone.size) { i ->
        floatArrayOf(  stereo_amp[0] * tone[i], stereo_amp[1] * tone[i]) // Mono to both channels
    }
    // val stereoBeep = makeStereoBeep(stereo, relAngle, freq = 880.0, duration = 0.12, amp = 0.5f)
    val remaining = interval - 0.12
    
    return Pair(stereo, remaining)
}

fun angleToRelative(userPos: Array<Float>, userHeadingDeg: Double, targetPos: Array<Float>): Double {
    val dx = targetPos[0] - userPos[0]
    val dy = targetPos[1] - userPos[1]
    val bearing = atan2(dy, dx) * 180 / PI
    val rel = (bearing - userHeadingDeg + 180) % 360 - 180
    return rel
}

fun angleToPan(relAngleDeg: Double): Double {
    val pan = 0.5 * (1 + relAngleDeg / 180.0)
    return pan.coerceIn(0.0, 1.0)
}

fun equalPowerGains(pan: Double): Pair<Double, Double> {
    val theta = pan * PI / 2
    val left = cos(theta)
    val right = sin(theta)
    val beta = 50.0
    
    val expLeft = exp(left * beta)
    val expRight = exp(right * beta)
    val sum = expLeft + expRight
    
    return Pair(expLeft / sum, expRight / sum)
}

fun applyItdStereo(
    leftBuf: FloatArray,
    rightBuf: FloatArray,
    relAngleDeg: Double,
    sr: Int = SR,
    maxItdMs: Double = 0.7
): Pair<FloatArray, FloatArray> {
    val maxDelayS = maxItdMs / 1000.0
    val delay = maxDelayS * sin(relAngleDeg * PI / 180)
    val delaySamples = (abs(delay) * sr).roundToInt()
    
    if (delaySamples == 0) {
        return Pair(leftBuf, rightBuf)
    }
    
    return if (delay > 0) {
        val newLeft = FloatArray(leftBuf.size)
        System.arraycopy(leftBuf, 0, newLeft, delaySamples, leftBuf.size - delaySamples)
        Pair(newLeft, rightBuf)
    } else {
        val newRight = FloatArray(rightBuf.size)
        System.arraycopy(rightBuf, 0, newRight, delaySamples, rightBuf.size - delaySamples)
        Pair(leftBuf, newRight)
    }
}

fun makeTone(freq: Double = 880.0, duration: Double = 0.12, sr: Int = SR, amp: Float = 0.6f): FloatArray {
    val samples = (sr * duration).toInt()
    val tone = FloatArray(samples)
    
    for (i in 0 until samples) {
        val t = i.toDouble() / sr
        tone[i] = (amp * sin(2 * PI * freq * t)).toFloat()
        
        // Apply Hanning window envelope
        val windowValue = 0.5 * (1 - cos(2 * PI * i / (samples - 1)))
        tone[i] *= windowValue.toFloat()
    }
    
    return tone
}

fun makeStereoBeep(
    gain: FloatArray,
    relAngleDeg: Double,
    freq: Double = 880.0,
    duration: Double = 0.12,
    amp: Float = 0.6f,
    sr: Int = SR
): Array<FloatArray> {
    val mono = makeTone(freq = freq, duration = duration, sr = sr, amp = amp)
    val lGain = gain[0]
    val rGain = gain[1]
    
    val left = mono.map { it * lGain }.toFloatArray()
    val right = mono.map { it * rGain }.toFloatArray()
    
    val (finalLeft, finalRight) = applyItdStereo(left, right, relAngleDeg, sr)
    
    // Convert to stereo format [sample][channel]
    val stereo = Array(finalLeft.size) { i ->
        floatArrayOf(finalLeft[i], finalRight[i])
    }
    
    return stereo
}

fun distanceToRate(
    distM: Double,
    minRate: Double = 0.5,
    maxRate: Double = 4.0,
    maxDist: Double = 20.0
): Double {
    val d = distM.coerceIn(0.0, maxDist)
    val rate = maxRate - d / maxDist * (maxRate - minRate)
    return rate
}

fun main() {
    val player = SoundPlayer()
    
    // Test 2: Directional sound
    println("\nTest 2: Playing directional sound...")
    var userPos = arrayOf(0.0f, 0.0f)
    val userHeading = 0.0
    val targetPos = arrayOf(5.0f, 5.0f)
    var t = 0
    while (true) {

        val (stereoSound, rate) = computeDirectionalSound(userPos, userHeading, targetPos)
        player.play(stereoSound, rate)
        // Thread.sleep(5)
        if (t > 2000) {
            userPos = arrayOf(5.0f, 5.0f)
        }
        t++
    }
    
    println("Sound tests complete!")
}