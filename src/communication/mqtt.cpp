#include <WiFi.h>
#include <PubSubClient.h>
#include "mqtt.h"
//  IP address of your computer running Mosquitto broker
const char* mqtt_server = "192.168.1.11";
//  WiFi client (network layer)
WiFiClient espClient;
//  MQTT client using WiFi connection
PubSubClient client(espClient);
/*
   reconnect()
  - Tries to reconnect ESP32 to MQTT broker if connection is lost
*/
void reconnect() {
    while (!client.connected()) {
        Serial.print("Connecting to MQTT...");
        //  Client ID = name of ESP32 in broker
        if (client.connect("ESP32Client")) {
            Serial.println("connected Successfully");
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            // wait before retry
            delay(2000);
        }
    }
}
/*
   mqttInit()
  - Sets MQTT server (broker IP + port)
*/
void mqttInit() {
    client.setServer(mqtt_server, 1883);
}
/*
   mqttLoop()
  - Keeps MQTT connection alive
  - MUST be called in main loop()
*/
void mqttLoop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop(); //  handles incoming/outgoing MQTT messages
}
/*
   mqttPublish()
  - Sends sensor data to MQTT broker
*/

void mqttPublish(float temp,
                 float voltage,
                 float current,
                 float vibration,
                 float humidity,
                 float pressure,
                 float power) {
    // ----------- Anomaly Detection -----------
    bool anomaly = false;
    if (vibration > 12.0) anomaly = true;   // High vibration
    if (voltage < 3.3) anomaly = true;      // Low voltage
    if (temp > 50.0) anomaly = true;        // High temperature
    // ----------- JSON Message -----------
    char msg[250];  // increased size (important)
    sprintf(msg,
        "{"
        "\"temp\":%.2f,"
        "\"voltage\":%.2f,"
        "\"current\":%.2f,"
        "\"vibration\":%.2f,"
        "\"humidity\":%.2f,"
        "\"pressure\":%.2f,"
        "\"power\":%.2f,"
        "\"anomaly\":%s"
        "}",
        temp, voltage, current, vibration,
        humidity, pressure, power,
        anomaly ? "true" : "false"
    );
    // ----------- Publish -----------
    client.publish("smartwind/data", msg);
    // ----------- Debug -----------
    Serial.print("Published: ");
    Serial.println(msg);
}