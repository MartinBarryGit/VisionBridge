import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.fasterxml.jackson.annotation.JsonProperty
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import java.util.Base64

import jakarta.validation.constraints.NotNull
import jakarta.validation.constraints.Size

data class Door(
    @JsonProperty("description")
    @field:NotNull
    @field:Size(min = 1, max = 500)
    val description: String,
    
    @JsonProperty("position")
    @field:NotNull
    val position: String, // "left", "right", "center"
    
    @JsonProperty("status")
    @field:NotNull
    val status: String, // "open", "closed"
    
    @JsonProperty("confidence")
    val confidence: Double? = null
)

data class DoorDetectionResponse(
    @JsonProperty("doors")
    @field:NotNull
    val doors: List<Door>,
    
    @JsonProperty("answer")
    @field:NotNull
    @field:Size(min = 1, max = 1000)
    val answer: String,
    
    @JsonProperty("status")
    @field:NotNull
    val status: Int // -1: no doors, 0: one door, 1: multiple doors
)


class AIDoorDescriptor {
    private val apiKey = System.getenv("OPENAI_API_KEY")
    private val client = HttpClient.newBuilder().build()
    private val objectMapper = jacksonObjectMapper()

    fun describeDoorWithStructure(image: BufferedImage): DoorDetectionResponse {
        if (apiKey.isNullOrEmpty()) {
            return DoorDetectionResponse(
                doors = emptyList(),
                answer = "Error: OpenAI API key not set.",
                status = -1
            )
        }
        return try {
            // Convert BufferedImage to base64
            val base64Image = bufferedImageToBase64(image)
            
            val prompt = """
                Analyze this image for doors and respond with a JSON object following this exact structure:
                
                {
                    "doors": [
                        {
                            "description": "Brief description of the door (material, color, features)",
                            "position": "left|right|center",
                            "status": "open|closed|unknown",
                            "confidence": 0.85
                        }
                    ],
                    "answer": "Human-readable response for visually impaired user describing doors and navigation guidance",
                    "status": 0
                }
                
                Status codes:
                - -1: No doors detected in the image
                - 0: One door found (ready to provide navigation guidance)
                - 1: Multiple doors found (user needs to choose)
                
                Look carefully at the image for:
                - Door frames, handles, hinges
                - Open or closed doors
                - Doorways and entrances
                - Position relative to the camera view (left, right, center)
                
                Provide helpful navigation guidance in the answer field, describing the door's location and how to approach it.
            """.trimIndent()
            
            val requestBody = """
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": ${escapeJsonString(prompt)}
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,$base64Image",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 300,
                    "response_format": { "type": "json_object" }
                }
            """.trimIndent()
            
            val request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.openai.com/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer $apiKey")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build()
            
            val response = client.send(request, HttpResponse.BodyHandlers.ofString())
            
            if (response.statusCode() == 200) {
                val responseBody = response.body()
                val choices = objectMapper.readTree(responseBody).get("choices")
                val message = choices.get(0).get("message")
                val content = message.get("content").asText()
                    .replace("\\n", "\n")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                
                println("Vision API Response: $content")
                objectMapper.readValue<DoorDetectionResponse>(content)
                
            } else {
                println("Vision API Error: ${response.statusCode()} - ${response.body()}")
                DoorDetectionResponse(
                    doors = emptyList(),
                    answer = "Error calling OpenAI Vision API: ${response.statusCode()}",
                    status = -1
                )
            }
        } catch (e: Exception) {
            println("Vision processing error: ${e.message}")
            e.printStackTrace()
            DoorDetectionResponse(
                doors = emptyList(),
                answer = "Error processing image: ${e.message}",
                status = -1
            )
        }
    }
    
    private fun bufferedImageToBase64(image: BufferedImage): String {
        return try {
            val outputStream = ByteArrayOutputStream()
            ImageIO.write(image, "jpg", outputStream)
            val imageBytes = outputStream.toByteArray()
            Base64.getEncoder().encodeToString(imageBytes)
        } catch (e: Exception) {
            println("Error converting image to base64: ${e.message}")
            ""
        }
    }
    
    private fun escapeJsonString(str: String): String {
        return "\"" + str
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t") + "\""
    }
    
    // Keep your original method for simple text-based responses
    // fun describeDoor(detectionInfo: String): String {
    //     if (apiKey.isNullOrEmpty()) {
    //         return "Error: OpenAI API key not set."
    //     }
        
    //     return try {
    //         val prompt = """
    //             Based on this door detection information, provide a helpful description for someone with visual impairment:
                
    //             $detectionInfo
                
    //             Please describe:
    //             1. The door's location and orientation
    //             2. Any notable features (handle, color, material if visible)
    //             3. Navigation guidance (how to approach the door)
                
    //             Keep the response concise and practical.
    //         """.trimIndent()
            
    //         val requestBody = """
    //             {
    //                 "model": "gpt-3.5-turbo",
    //                 "messages": [
    //                     {
    //                         "role": "user",
    //                         "content": ${escapeJsonString(prompt)}
    //                     }
    //                 ],
    //                 "max_tokens": 150
    //             }
    //         """.trimIndent()
            
    //         val request = HttpRequest.newBuilder()
    //             .uri(URI.create("https://api.openai.com/v1/chat/completions"))
    //             .header("Content-Type", "application/json")
    //             .header("Authorization", "Bearer $apiKey")
    //             .POST(HttpRequest.BodyPublishers.ofString(requestBody))
    //             .build()
            
    //         val response = client.send(request, HttpResponse.BodyHandlers.ofString())
            
    //         if (response.statusCode() == 200) {
    //             val responseBody = response.body()
    //             val choices = objectMapper.readTree(responseBody).get("choices")
    //             val message = choices.get(0).get("message")
    //             val content = message.get("content").asText()
                
    //             content
    //         } else {
    //             "Error calling OpenAI API: ${response.statusCode()}"
    //         }
    //     } catch (e: Exception) {
    //         "Error: ${e.message}"
    //     }
    // }
    
    // Image-based description method
    fun describeDoor(image: BufferedImage): DoorDetectionResponse {
        val structuredResponse = describeDoorWithStructure(image)
        return structuredResponse
    }
}

fun main() {
    val aiDescriptor = AIDoorDescriptor()
    val sampleImage = BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB)
    val description = aiDescriptor.describeDoor(sampleImage)
    println("AI Description: $description")
}