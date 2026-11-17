# How our exam program works (explain this in the oral check)

1. **Load the model first.** `exam.py` loads our trained YOLO26 model (`best.pt`) and runs it once on a black image to warm it up, so the first real prediction isn't slow.
2. **Connect to the Tello.** It connects over the Tello Wi-Fi, prints the battery, and turns on the live video stream.
3. **Every video frame:** it converts the colours from RGB to BGR (what OpenCV uses) and runs our model on it.
4. **Draw the result.** It takes the most confident detection and draws a box with the teammate's name and the **raw** confidence YOLO gave, e.g. `memberB 0.912`. We never change that number.
5. **Fly with the keyboard.** The pilot sees only this window. Keys send `send_rc_control` commands; when no key is pressed it sends 0,0,0,0 so the drone hovers. SPACE = hover, l = land, x = emergency.
6. **Avoid wrong names.** It only saves once the **same name** appears in 5 frames in a row (with confidence ≥ 0.50), so one random mistake can't trigger a save.
7. **Pick the clearest frame.** For 1 second it keeps the frame with the highest raw confidence, then saves that photo to `exam_photos/` with the name, confidence and time in the filename.
8. **Check the photo.** It reopens the saved file to make sure it's a valid image, then shows `SAVED OK` on screen. If it failed, press `p` to save manually.
9. **Safety at the end.** Whatever happens (even an error), the `finally` block lands the drone and closes the video.

## How the model was made
- `03_collect_photos.py`: we took ~100 photos of each teammate with the **Tello camera** in the classroom, plus empty-room photos.
- `04_make_dataset.py`: the pretrained `yolo26n.pt` (which already knows "person") drew a box around the person in each photo; the class (name) comes from the folder we saved it in. We checked the boxes in X-AnyLabeling. 80% train / 20% val.
- `05_train.py`: trained YOLO26n (starting from the pretrained `yolo26n.pt`) on our photos.

**Tip:** the model learns each teammate from face, hair and clothes, so wear the same clothes on exam day as when collecting photos.
