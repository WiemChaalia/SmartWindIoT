#ifndef MQTT_H
#define MQTT_H

void mqttInit();
void mqttLoop();
void mqttPublish(float temp,
                  float voltage,
                  float current,
                  float vibration,
                  float humidity,
                  float pressure,
                  float power);

#endif