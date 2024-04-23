#include "ArduinoJson.h"
#define IN1_PIN   5   
#define IN2_PIN   6   
#define IN3_PIN   9   
#define IN4_PIN   10    

StaticJsonDocument<200> jsondoc;

int16_t speedA = 0;
int16_t speedB = 0;


void setup()
{
  Serial.begin(9600);
  pinMode(IN1_PIN, OUTPUT);    
  pinMode(IN2_PIN, OUTPUT);    
  pinMode(IN3_PIN, OUTPUT);    
  pinMode(IN4_PIN, OUTPUT);    
}

void loop()
{
  DeserializationError err = deserializeJson(jsondoc, Serial);  
  if (err == DeserializationError::Ok)   
  {
    speedA = (float)jsondoc["speedA"] * 2.55;
    speedB = (float)jsondoc["speedB"] * 2.55;
    mode = (int)jsondoc["mode"];
    iSee = (bool)jsondoc["iSee"];
  }
  else
  {
    while (Serial.available() > 0) Serial.read();
  }

  if (mode == 2){
    moveA(speedA);
    moveB(speedB);
  } else {
    if (iSee == False){
      speedA = 0;
      speedB = 0;
    }
    moveA(speedA);
    moveB(speedB);
  }

  delay(10);
}

void moveA(int16_t speed){
  //speed = -speed;
  if(speed >= 0){
    analogWrite(IN1_PIN, LOW);  
    analogWrite(IN2_PIN, abs(speed));
  } 
  else {
    analogWrite(IN1_PIN, abs(speed));  
    analogWrite(IN2_PIN, LOW);
  }
}

void moveB(int16_t speed){
  if(speed >= 0){
    analogWrite(IN3_PIN, LOW);  
    analogWrite(IN4_PIN, abs(speed));
  } 
  else {
    analogWrite(IN3_PIN, abs(speed));  
    analogWrite(IN4_PIN, LOW);
  }
}
 