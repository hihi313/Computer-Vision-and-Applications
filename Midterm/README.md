# Image preprocessing

Scale down to increase processing speed

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L30-L31

## Denoise

1. Get the highest contrast channel (luminance channel)
    * Convert to CIELAB format
2, Median blur & morphological open & close to remove noise

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L49-L60

## Get contour

1. Use Canny to detect edges

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L78-L82

2. Find contours

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L109-L110

# Extract A4 sheet 

