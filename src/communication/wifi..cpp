#include <WiFi.h>

const char* ssid = "TT_4BFF_2.4G";
const char* password = "Tnyes4XC4n";

void connectWiFi() {
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }

    Serial.println("WiFi connected");
}