// Arduino Supervisory Controller (concept firmware)
// Responsibilities:
//  - Read bus voltage/current (INA219/INA260 or similar) -> available power estimate
//  - Enforce: hard cap 25W, nominal target 75% of available
//  - Send telemetry + commands to Jetson over UART/I2C/CAN
//
// NOTE: This sketch is a template. Sensor wiring + calibration constants are mission-specific.

#include <Arduino.h>

// ---- User settings ----
static const float POWER_CAP_W = 25.0f;
static const float NOMINAL_FRACTION = 0.75f;

// Telemetry send period
static const uint32_t TELEMETRY_MS = 500;

// Example pins (replace with INA219 library reads)
static const int PIN_VOLT = A0;
static const int PIN_CURR = A1;

uint32_t lastTx = 0;

float readBusVoltage_V(){
  // Stub: convert ADC reading to volts (replace with real sensor)
  int raw = analogRead(PIN_VOLT);
  return (raw / 1023.0f) * 5.0f * 20.0f; // example divider
}
float readBusCurrent_A(){
  int raw = analogRead(PIN_CURR);
  return (raw / 1023.0f) * 5.0f; // example scale
}

void setup(){
  Serial.begin(115200);
  analogReference(DEFAULT);
}

void loop(){
  float V = readBusVoltage_V();
  float I = readBusCurrent_A();
  float availableW = V * I;         // simplistic: available to payload bus (replace with EPS-provided value)
  float targetW = min(POWER_CAP_W, availableW * NOMINAL_FRACTION);

  // In flight: also read Jetson draw (or estimate) and temperature sensors
  float jetsonDrawW = 0.0f; // unknown here

  uint32_t now = millis();
  if(now - lastTx >= TELEMETRY_MS){
    lastTx = now;
    // Send JSON-like telemetry line the Jetson can parse
    Serial.print("{\"available_W\":");
    Serial.print(availableW, 2);
    Serial.print(",\"target_W\":");
    Serial.print(targetW, 2);
    Serial.print(",\"cap_W\":");
    Serial.print(POWER_CAP_W, 2);
    Serial.println("}");
  }

  // Decision example: if targetW is low, command Jetson to reduce workload
  // Serial.println("CMD:SET_MODE LOWRES"); etc.

  delay(20);
}
