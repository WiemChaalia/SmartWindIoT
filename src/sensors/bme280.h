#ifndef BME280_H
#define BME280_H

// Initialisation du capteur
void initBME();

// Fonctions de lecture
float getTemperature();
float getHumidity();
float getPressure();

#endif