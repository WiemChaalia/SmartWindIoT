#include <Wire.h>
#include <Adafruit_BME280.h>

// Création de l'objet capteur
Adafruit_BME280 bme;

// Initialisation du capteur
void initBME() {
    // Démarrage du bus I2C (SDA=21, SCL=22 pour ESP32)
    Wire.begin(21, 22);

    // Adresse I2C du BME280 (souvent 0x76 ou 0x77)
    if (!bme.begin(0x76)) {
        Serial.println(" BME280 non détecté !");
        while (1); // Bloque si erreur
    }

    Serial.println(" BME280 initialisé");
}

// Lecture température en °C
float getTemperature() {
    return bme.readTemperature();
}

// Lecture humidité en %
float getHumidity() {
    return bme.readHumidity();
}

// Lecture pression en hPa
float getPressure() {
    return bme.readPressure() / 100.0F;
}