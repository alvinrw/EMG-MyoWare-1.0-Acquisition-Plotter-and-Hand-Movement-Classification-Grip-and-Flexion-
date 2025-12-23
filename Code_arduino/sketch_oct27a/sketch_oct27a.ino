int sensorPin = 34;
int sensorValue = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("MyoWare EMG Sensor - ESP32");
}

void loop() {
  sensorValue = analogRead(sensorPin);
  Serial.println(sensorValue);
  delay(100);
}
