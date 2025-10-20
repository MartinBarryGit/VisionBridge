plugins {
    kotlin("jvm") version "2.2.20"
    application
}

group = "com.visionbridge"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    // Jackson for JSON parsing (Pydantic equivalent)
    implementation("com.fasterxml.jackson.core:jackson-core:2.15.2")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.15.2")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.15.2")
    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.15.2")
    
    // Validation (JSR-303/Bean Validation)
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("org.hibernate.validator:hibernate-validator:8.0.1.Final")
    implementation("org.glassfish.expressly:expressly:5.0.0")
    
    // ONNX Runtime for door detection
    implementation("com.microsoft.onnxruntime:onnxruntime:1.16.3")
    
    // OpenCV (optional, if you need it)
    implementation(files("libs/opencv-4.7.0-0.jar"))
}

sourceSets {
    main {
        kotlin {
            srcDirs("./utils")  // Use current directory as source
        }
    }
}

application {
    mainClass.set("DoorDetectionFunctionsKt") // Set your main class here
}

// Configure both Java and Kotlin to use compatible versions
tasks.withType<JavaCompile> {
    targetCompatibility = "21"
    sourceCompatibility = "21"
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_21)
    }
} 

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}