
import webbrowser
import os

# Define the path to your generated HTML file
file_path = 'interactive_facebook_network1.html'

# Convert the relative path to an absolute path
abs_file_path = os.path.abspath(file_path)

# Open the file in the default web browser
webbrowser.open('file://' + abs_file_path)
