#!/usr/bin/env python3
"""
Quick utility to test macOS Accessibility permissions and verify that
simulated Enter/Return keypresses are received by active applications.
"""

import sys
import time
import pyautogui

from config import IS_MACOS

print("=" * 65)
print("  CROSS-PLATFORM KEYPRESS VERIFICATION TOOL")
print("=" * 65)
print("Instructions:")
print("  1. Immediately after launching, click on any target app with")
print("     a text cursor (e.g., Browser search bar, Notes, or a text box).")
print("  2. In 3 seconds, this script will simulate pressing Enter (Return).")
print("=" * 65)

for i in range(3, 0, -1):
    print(f"Emulating keypress in {i}...")
    time.sleep(1)

print("\n>>> DISPATCHING ENTER / RETURN KEYPRESS NOW <<<")

if IS_MACOS:
    # On macOS, Cocoa / GUI inputs expect 'return' (code 36)
    pyautogui.press("return")
else:
    pyautogui.press("enter")

print("\n[VERIFICATION RESULT]")
print("If your focused app registered the line break, search submit, or form submit,")
print("then PyAutoGUI OS permissions and key codes are working properly!")
print("=" * 65)
