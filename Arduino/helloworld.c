#include "Keyboard.h"

void setup() {
  Keyboard.begin();
}

void loop() {
  delay(5000);
  Keyboard.print("hello, world!");
  delay(500);
  Keyboard.press(KEY_CAPS_LOCK);
  delay(100);
  Keyboard.releaseAll();
  // do nothing:
  while (true);
}
