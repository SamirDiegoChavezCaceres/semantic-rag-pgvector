# Generate the demo GIF with VHS:  vhs demo.tape
# Install VHS (one binary): https://github.com/charmbracelet/vhs
#
# On Windows you may need to tell VHS which shell to use, e.g.:
#   Set Shell "powershell"

Output assets/demo.gif

Set FontSize 16
Set Width 1100
Set Height 640
Set Padding 20

# Install the package quietly, off-camera.
Hide
Type "pip install -e . -q" Enter
Sleep 12s
Ctrl+L
Show

Type "python scripts/demo.py" Enter
Sleep 6s
