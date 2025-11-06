int sensorPin =34;
int sensorValue = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("Data dari MyoWare - esp");
}

void loop() {
  sensorValue = analogRead(sensorPin);
  Serial.println(sensorValue);
  delay(100);
}
