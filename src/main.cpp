#include <Arduino.h>

#include "communication/wifi.h"
#include "communication/mqtt.h"
#include "sensors/bme280.h"
#include "sensors/ina219.h"
#include "sensors/mpu6050.h"

void setup() {
    Serial.begin(115200);

    connectWiFi();
    mqttInit();

    initBME();
    initINA();
    initMPU();
}

void loop() {
    float temp = getTemperature();
    float voltage = getVoltage();
    float current = getCurrent();
    float vibration = getVibration();
    float humidity = getHumidity();
    float pressure = getPressure();
    float power = getPower();

    Serial.println("===== SENSOR DATA =====");
    Serial.print("Temp: "); Serial.println(temp);
    Serial.print("Voltage: "); Serial.println(voltage);
    Serial.print("Current: "); Serial.println(current);
    Serial.print("Vibration: "); Serial.println(vibration);
    Serial.print("Humidity: "); Serial.println(humidity);
    Serial.print("Pressure: "); Serial.println(pressure);
    Serial.print("Power: "); Serial.println(power);


    mqttPublish(temp, voltage, current, vibration, humidity, pressure, power);

    delay(2000);
}