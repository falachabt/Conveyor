#include <Wire.h>
#include "Adafruit_TCS34725.h"

// Color sensor configuration
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_1X);

// Pin definitions
#define PIN_DIR1 5
#define PIN_STEP1 2
#define PIN_DIR2 6
#define PIN_STEP2 3

// Sensor pins
#define ENTRY_SENSOR 9      // First conveyor entry IR sensor
#define EXIT_SENSOR 10      // First conveyor exit IR sensor
#define BOX_LEFT_LIMIT 11   // Second conveyor left limit IR sensor
#define BOX_RIGHT_LIMIT 12  // Second conveyor right limit IR sensor

// Motor control constants
int STEP_DELAY_MICROS = 300;  // Changed from const to variable for speed control
const int STEPS_PER_CYCLE = 500;
const int END_STEP_DELAY = 1000;

// Color definitions
#define NONE 0
#define BLUE 1
#define BLACK 2
#define RED 3
#define YELLOW 4

// State machine states for Conveyor 1
#define C1_IDLE 0
#define C1_RUNNING 1
#define C1_WAIT_DROP 2
#define C1_DROPPING 3

// State machine states for Conveyor 2
#define C2_IDLE 0
#define C2_FINDING_BOX 1
#define C2_POSITIONING 2
#define C2_READY 3

// System variables
int state_C1 = C1_IDLE;
int old_state_C1 = C1_IDLE;
int state_C2 = C2_IDLE;
int old_state_C2 = C2_IDLE;

int target_color = BLUE;  // Default target color
int current_box_color = NONE;
int objectCount = 0;
bool box_positioned = false;
bool drop_complete = false;
bool color_search_enabled = false;  // Flag to control when to search for color

// Direction control for the second conveyor
bool c2_direction_right = true;  // true = right movement, false = left movement

unsigned long lastEntryTime = 0;
const int ENTRY_TIMEOUT_MS = 1000;

// Serial command handling
String inputString = "";
boolean stringComplete = false;

// Helper functions
String getColorName(int color) {
  switch (color) {
    case RED: return "RED";
    case BLUE: return "BLUE";
    case BLACK: return "BLACK";
    case YELLOW: return "YELLOW";
    default: return "NONE";
  }
}

int readColorSensor() {
  if (!color_search_enabled) {
    return NONE;
  }

  uint16_t r, g, b, c;
  tcs.getRawData(&r, &g, &b, &c);

  // Convert to percentages
  unsigned int R = r * 100 / c;
  unsigned int G = g * 100 / c;
  unsigned int B = b * 130 / c;

  // Print raw color values
  Serial.print("Color Reading - R: ");
  Serial.print(R);
  Serial.print(", G: ");
  Serial.print(G);
  Serial.print(", B: ");
  Serial.println(B);

  // Box color detection thresholds
  // BLACK detection (R~38, G~35, B~34-35)
  if (R >= 35 && R <= 42 && G >= 32 && G <= 40 && B >= 30 && B <= 40) {
    Serial.println("Detected: BLACK");
    return BLACK;
  }
  // BLUE detection (R~30-31, G~39, B~48-50)
  else if (R >= 25 && R <= 35 && G >= 35 && G <= 45 && B >= 45 && B <= 55) {
    Serial.println("Detected: BLUE");
    return BLUE;
  }
  // RED detection (R~65-75, G~19-23, B~23-27)
  else if (R >= 55 && R <= 80 && G >= 15 && G <= 30 && B >= 20 && B <= 35) {
    Serial.println("Detected: RED");
    return RED;
  }
  // YELLOW detection (assuming from typical values)
  else if (R >= 50 && R <= 80 && G >= 50 && G <= 80 && B >= 15 && B <= 35) {
    Serial.println("Detected: YELLOW");
    return YELLOW;
  } else {
    Serial.println("Detected: NONE");
    return NONE;
  }
  Serial.println("Detected: NONE");
  return NONE;
}

void runMotorStep(int stepPin, int delayTime) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(delayTime);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(delayTime);
}

void setConveyor2Direction(bool goRight) {
  if (goRight) {
    digitalWrite(PIN_DIR2, LOW);  // Corrected: Move right (assuming this is the correct pin logic)
    c2_direction_right = true;
    Serial.println("Conveyor 2 direction: RIGHT");
  } else {
    digitalWrite(PIN_DIR2, HIGH);  // Corrected: Move left
    c2_direction_right = false;
    Serial.println("Conveyor 2 direction: LEFT");
  }
}

void invertConveyor2Direction() {
  setConveyor2Direction(!c2_direction_right);
}

void setup() {
  Serial.begin(9600);

  // Configure pins
  pinMode(PIN_DIR1, OUTPUT);
  pinMode(PIN_STEP1, OUTPUT);
  pinMode(PIN_DIR2, OUTPUT);
  pinMode(PIN_STEP2, OUTPUT);

  pinMode(ENTRY_SENSOR, INPUT);
  pinMode(EXIT_SENSOR, INPUT);
  pinMode(BOX_LEFT_LIMIT, INPUT);
  pinMode(BOX_RIGHT_LIMIT, INPUT);

  // Initialize color sensor
  if (!tcs.begin()) {
    Serial.println("No TCS34725 found... check your connections");
    while (1)
      ;
  }

  // Initialize motors
  digitalWrite(PIN_DIR1, LOW);  // Set first conveyor direction
  setConveyor2Direction(true);  // Initialize second conveyor to move right
  digitalWrite(PIN_STEP1, HIGH);
  digitalWrite(PIN_STEP2, HIGH);

  inputString.reserve(200);
  Serial.println("READY");
  Serial.println("System Ready");
}

void conveyor1_StateMachine() {
  old_state_C1 = state_C1;

  switch (state_C1) {
    case C1_IDLE:
      // In idle state, we're just waiting for a new object to be detected at the entry sensor
      if (!digitalRead(ENTRY_SENSOR) && (millis() - lastEntryTime > ENTRY_TIMEOUT_MS)) {
        state_C1 = C1_RUNNING;
        lastEntryTime = millis();
        objectCount++;
        Serial.println("NEW_OBJECT_DETECTED");  // Signal Python that object is detected
        Serial.println("Object detected! Moving object on conveyor...");
      }
      break;

    case C1_RUNNING:
      // Continue running conveyor 1 until reaching exit sensor
      for (int i = 0; i < STEPS_PER_CYCLE; i++) {
        runMotorStep(PIN_STEP1, STEP_DELAY_MICROS);

        // Check if any new objects entered
        if (!digitalRead(ENTRY_SENSOR) && (millis() - lastEntryTime > ENTRY_TIMEOUT_MS)) {
          objectCount++;
          lastEntryTime = millis();
          Serial.println("Additional object detected during movement!");
        }

        // Check if we've reached the exit sensor
        if (!digitalRead(EXIT_SENSOR)) {
          state_C1 = C1_WAIT_DROP;
          color_search_enabled = true;  // Enable color search when object reaches exit
          Serial.println("Object at exit position. Waiting for box positioning.");
          break;
        }
      }
      break;

    case C1_WAIT_DROP:
      // Check for new objects while waiting
      if (!digitalRead(ENTRY_SENSOR) && (millis() - lastEntryTime > ENTRY_TIMEOUT_MS)) {
        objectCount++;
        lastEntryTime = millis();
        Serial.println("NEW_OBJECT_DETECTED");  // Signal Python for next object
      }

      // Wait for conveyor 2 to be ready with the correct color box positioned
      if (state_C2 == C2_READY) {
        Serial.println("Starting drop sequence...");
        // Run conveyor 1 for longer to ensure object drops
        for (int i = 0; i < STEPS_PER_CYCLE; i++) {
          runMotorStep(PIN_STEP1, STEP_DELAY_MICROS);
        }
        // Add a second cycle of movement for more reliable dropping
        for (int i = 0; i < STEPS_PER_CYCLE; i++) {
          runMotorStep(PIN_STEP1, STEP_DELAY_MICROS);
        }
        // Third step to make sure the object drops
        for (int i = 0; i < STEPS_PER_CYCLE; i++) {
          runMotorStep(PIN_STEP1, STEP_DELAY_MICROS);
        }
        state_C1 = C1_DROPPING;
        drop_complete = false;
        Serial.println("Drop started, waiting for completion...");
      }
      break;

    case C1_DROPPING:
      // Object has been dropped
      drop_complete = true;
      objectCount--;

      Serial.println("Drop complete. Color: " + getColorName(target_color));

      // Check if another object is already at exit
      if (!digitalRead(EXIT_SENSOR)) {
        state_C1 = C1_WAIT_DROP;
        Serial.println("Another object already at exit position.");
      }
      // Otherwise check if more objects are in the queue
      else if (objectCount > 0) {
        state_C1 = C1_RUNNING;
        Serial.println("Continuing with next object");
      } else {
        state_C1 = C1_IDLE;
        Serial.println("No more objects, returning to idle");
      }

      // Reset conveyor 2 state
      state_C2 = C2_IDLE;
      color_search_enabled = false;
      break;
  }
}

void conveyor2_StateMachine() {
  old_state_C2 = state_C2;

  // Check limit sensors for direction changes regardless of state
  if (!digitalRead(BOX_LEFT_LIMIT) && !c2_direction_right) {
    // Hit left limit while moving left, change to right
    invertConveyor2Direction();
    Serial.println("Left limit triggered. Inverting direction.");
  } else if (!digitalRead(BOX_RIGHT_LIMIT) && c2_direction_right) {
    // Hit right limit while moving right, change to left
    invertConveyor2Direction();
    Serial.println("Right limit triggered. Inverting direction.");
  }

  switch (state_C2) {
    case C2_IDLE:
      if (state_C1 == C1_WAIT_DROP && color_search_enabled) {
        state_C2 = C2_FINDING_BOX;
        current_box_color = NONE;  // Reset color detection
        Serial.println("Starting to look for " + getColorName(target_color) + " box");
      }
      break;

    case C2_FINDING_BOX:
      current_box_color = readColorSensor();
      if (current_box_color == target_color) {
        Serial.println(getColorName(target_color) + " box found! Moving to positioning...");
        state_C2 = C2_POSITIONING;
      } else {
        // Run the conveyor in the current direction until we find the right box
        for (int i = 0; i < STEPS_PER_CYCLE; i++) {
          runMotorStep(PIN_STEP2, STEP_DELAY_MICROS);

          // Recheck color every few steps
          if (i % 50 == 0) {
            current_box_color = readColorSensor();
            if (current_box_color == target_color) {
              Serial.println(getColorName(target_color) + " box found during movement!");
              state_C2 = C2_POSITIONING;
              break;
            }
          }

          // Check limit sensors during movement
          if (!digitalRead(BOX_LEFT_LIMIT) && !c2_direction_right) {
            invertConveyor2Direction();
            Serial.println("Left limit triggered during search. Inverting direction.");
            break;
          } else if (!digitalRead(BOX_RIGHT_LIMIT) && c2_direction_right) {
            invertConveyor2Direction();
            Serial.println("Right limit triggered during search. Inverting direction.");
            break;
          }
        }
      }
      break;

    case C2_POSITIONING:
      Serial.println("Fine positioning for 1 second...");
      unsigned long startTime = millis();
      while (millis() - startTime < 1000) {
        runMotorStep(PIN_STEP2, END_STEP_DELAY);

        // Check limit sensors even during fine positioning
        if (!digitalRead(BOX_LEFT_LIMIT) && !c2_direction_right) {
          invertConveyor2Direction();
          Serial.println("Left limit triggered during positioning. Inverting direction.");
          break;
        } else if (!digitalRead(BOX_RIGHT_LIMIT) && c2_direction_right) {
          invertConveyor2Direction();
          Serial.println("Right limit triggered during positioning. Inverting direction.");
          break;
        }
      }
      state_C2 = C2_READY;
      Serial.println("Box positioned! Ready for drop.");
      break;

    case C2_READY:
      if (drop_complete) {
        state_C2 = C2_IDLE;
      }
      break;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    // Add character to the string
    if (inChar != '\n' && inChar != '\r') {
      inputString += inChar;
    }
    // Process the string when new line or carriage return arrives
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    }
  }
}

void processSerialCommand(String command) {
  command.trim();

  if (command == "1" || command == "2" || command == "3") {
    // Get the class number
    int classNumber = command.toInt();

    Serial.println("received command" + command);

    // Set the target color based on the class
    switch (classNumber) {
      case 1:
        target_color = BLUE;
        break;
      case 2:
        target_color = BLACK;
        break;
      case 3:
        target_color = RED;
        break;
      default:
        target_color = RED;
        break;
    }

    Serial.println("Received class " + command + " (" + getColorName(target_color) + ")");
  } else if (command == "STOP") {
    state_C1 = C1_IDLE;
    state_C2 = C2_IDLE;
    Serial.println("System stopped");
  } else if (command == "SLOW") {
    // Implement slow mode by changing step delay
    STEP_DELAY_MICROS = 800;  // Slower speed
    Serial.println("Slow mode activated");
  } else if (command == "NORMAL") {
    // Restore normal speed
    STEP_DELAY_MICROS = 400;  // Normal speed
    Serial.println("Normal speed activated");
  } else if (command == "RESET") {
    // Reset the system
    state_C1 = C1_IDLE;
    state_C2 = C2_IDLE;
    objectCount = 0;
    Serial.println("System reset");
  } else if (command == "DIR_LEFT") {
    // Force conveyor 2 direction to left
    setConveyor2Direction(false);
  } else if (command == "DIR_RIGHT") {
    // Force conveyor 2 direction to right
    setConveyor2Direction(true);
  } else if (command == "STATUS") {
    Serial.println("C1 State: " + String(state_C1));
    Serial.println("C2 State: " + String(state_C2));
    Serial.println("Objects: " + String(objectCount));
    Serial.println("Target Color: " + getColorName(target_color));
    Serial.println("Current Speed: " + String(STEP_DELAY_MICROS));
    Serial.println("C2 Direction: " + String(c2_direction_right ? "RIGHT" : "LEFT"));
  } else {
    Serial.println("Unknown command: " + command);
  }
}

void loop() {
  if (stringComplete) {
    processSerialCommand(inputString);
    inputString = "";
    stringComplete = false;
  }

  conveyor1_StateMachine();
  conveyor2_StateMachine();

  delay(10);  // Small delay for stability
}