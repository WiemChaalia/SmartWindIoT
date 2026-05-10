#include <Wire.h>
#include <Adafruit_INA219.h>

// Création de l'objet INA219
Adafruit_INA219 ina219;

// Initialisation du capteur
void initINA() {
    // Démarrage du capteur
    if (!ina219.begin()) {
        Serial.println(" INA219 non détecté !");
        while (1);
    }

    Serial.println(" INA219 initialisé");
}

// Lecture tension (Bus Voltage)
float getVoltage() {
    return ina219.getBusVoltage_V();
}

// Lecture courant (mA)
float getCurrent() {
    return ina219.getCurrent_mA();
}

// Calcul puissance (P = U * I)
float getPower() {
    return getVoltage() * (getCurrent() / 1000.0); // en Watts
}