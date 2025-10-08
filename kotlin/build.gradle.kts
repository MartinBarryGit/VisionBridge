plugins {
    kotlin("jvm") version "1.9.20"
    application
}

group = "com.visionbridge"
version = "1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("com.microsoft.onnxruntime:onnxruntime:1.16.3")
}

application {
    mainClass.set("Door_detection_functionsKt")
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions.jvmTarget = "11"
}

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}
