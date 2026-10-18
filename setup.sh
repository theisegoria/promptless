#!/bin/bash

# Folder containing this script, so it can be run from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define farewell and greeting lists
farewells=(
    "Goodbye! Take care!"
    "Farewell, my friend."
    "See you soon!"
    "Until we meet again!"
    "Wishing you all the best!"
    "Stay safe and take care!"
    "It was great knowing you!"
    "Good luck on your journey!"
    "Parting is such sweet sorrow."
    "Catch you later!"
    "Bye for now!"
    "May our paths cross again!"
    "Take care and keep in touch!"
    "So long, and thanks for everything!"
    "Adieu, until next time!"
    "Keep shining! Farewell!"
    "I'll miss you! Stay well."
    "Time to say goodbye. Be happy!"
    "See you in another life, maybe!"
    "The end of one journey is the start of another!"
    "[BYE]"
)

greetings=(
    "Hello! Nice to see you!"
    "Welcome back!"
    "Good to have you here!"
    "Long time no see!"
    "Greetings, my friend!"
    "Hey there! How have you been?"
    "Nice to meet you!"
    "Hope you're doing well!"
    "Welcome aboard!"
    "A pleasure to see you again!"
    "Hi! What's new?"
    "Good to see you!"
    "Happy to have you here!"
    "It’s been a while! Welcome!"
    "Glad to see you back!"
    "How’s everything going?"
    "Hope you're having a great day!"
    "Welcome home!"
    "What a wonderful surprise to see you!"
    "Here’s to new beginnings!"
    "Sup"
)

echo "Creating StopButton . . ."
add_command=("python" "$SCRIPT_DIR/__main__.py" "add" "--name" "StopButton" "--target" "${farewells[@]}" "--contrast" "${greetings[@]}")
"${add_command[@]}"

echo "Finished setting up StopButton. Currently stored states:"
view_command=("python" "$SCRIPT_DIR/__main__.py" "view")
"${view_command[@]}"