
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Création de l'objet MPU6050
Adafruit_MPU6050 mpu;

// Initialisation du capteur
void initMPU() {
    if (!mpu.begin()) {
        Serial.println(" MPU6050 non détecté !");
        while (1);
    }

    Serial.println(" MPU6050 initialisé");
}

// Calcul vibration (norme du vecteur accélération)
float getVibration() {
    sensors_event_t a, g, temp;

    // Lecture des données du capteur
    mpu.getEvent(&a, &g, &temp);

    // Calcul de la magnitude (sqrt(x² + y² + z²))
    float vibration = sqrt(
        a.acceleration.x * a.acceleration.x +
        a.acceleration.y * a.acceleration.y +
        a.acceleration.z * a.acceleration.z
    );

    return vibration;
}