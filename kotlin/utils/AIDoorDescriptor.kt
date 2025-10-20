import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.fasterxml.jackson.annotation.JsonProperty

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
    
    fun describeDoorWithStructure(detectionInfo: String): DoorDetectionResponse {
        if (apiKey.isNullOrEmpty()) {
            return DoorDetectionResponse(
                doors = emptyList(),
                answer = "Error: OpenAI API key not set.",
                status = -1
            )
        }
        println("Using API Key: $apiKey")
        return try {
            val prompt = """
                Analyze this door detection information and respond with a JSON object following this exact structure:
                
                {
                    "doors": [
                        {
                            "description": "Brief description of the door",
                            "position": "left|right|center",
                            "status": "open|closed|unknown",
                            "confidence": 0.85
                        }
                    ],
                    "answer": "Human-readable response for visually impaired user",
                    "status": 0
                }
                
                Status codes:
                - -1: No doors detected
                - 0: One door found (ready to guide)
                - 1: Multiple doors found (user needs to choose)
                
                Detection info: $detectionInfo
                
                Provide helpful navigation guidance in the answer field.
            """.trimIndent()
            
            val requestBody = """
                {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "user",
                            "content": ${escapeJsonString(prompt)}
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
                val content = message.get("content").asText().replace("\\n", "\n")
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                objectMapper.readValue<DoorDetectionResponse>(content)
                
            } else {
                DoorDetectionResponse(
                    doors = emptyList(),
                    answer = "Error calling OpenAI API: ${response.statusCode()}",
                    status = -1
                )
            }
        } catch (e: Exception) {
            DoorDetectionResponse(
                doors = emptyList(),
                answer = "Error: ${e.message}",
                status = -1
            )
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
    
    // Keep your original method for simple responses
    fun describeDoor(detectionInfo: String): String {
        val structuredResponse = describeDoorWithStructure(detectionInfo)
        return structuredResponse.answer
    }
}

fun main() {
    val aiDescriptor = AIDoorDescriptor()
    val sampleDetection = "Door detected at coordinates (150, 200, 300, 400) with confidence 0.85. Door appears to be wooden, brown color, located in the center-left of the image."
    val description = aiDescriptor.describeDoor(sampleDetection)
    println("AI Description: $description")
}