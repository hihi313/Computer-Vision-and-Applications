# Image preprocessing

Scale down to increase processing speed.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L30-L31

## Denoise

1. Get the highest contrast channel (luminance channel).
    * Convert to CIELAB format.
2, Median blur & morphological open & close to remove noise.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L49-L60

## Get contour

1. Use Canny to detect edges.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L78-L82

2. Find contours.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/e325914f254cc6adf1c8c7bd6b2ea6b902fb12d5/Midterm/feet_measure.py#L109-L110

# Extract A4 sheet 

## Get max convex hull

Using convex hull to eliminate the occlusion (by the foot).
Then it's reasonable that the A4 sheet will have the largest (hull) area in the images.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L121

1.  Get quadrilateral.

Approximate all the obtained contours & find the quadrilateral (4-points polygonal), to filter out most unlikely shape. 

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L141-L144

2.  Get max hull.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L159-L162

## Homography transform

1.  Make these corners ordered.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L170-L184

2. Get destination corners.
    * Once the destination size is obtained by given PPI, then we can get the destination corners directly.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L195-L202

3. Homography transform.
    * Using the edge image, because it's good representation that eliminate disturbances of color pixels.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L217-L222

# Extract foot

Using non-foot region to extract foot's contour, because OpenCV will treat extract the foot edges/lines as contour.

1. Get non-foot region.
    * Select the contour that is corresponding to the largest convex hull.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L234-L237

2. Get foot image by 1 - non-foot region.

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L239-L243

3. Find foot contour from previous image.
    * And it should now be the largest contour

https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L248-L251

# Combine feets from 2 views into a foot

Combine 2 views into a foot by top & bottom points of foot in both view

## Get top & bottom points from both views

To eliminate the disturbance of leg's contour points when extract top & bottom points based on the y axis value. Using angle to do this.

The same procedures apply to both top & bottom point. Only different from their corresponding angles.

Angle: 
```
   270
180   0
    90
```

1. Get points whithin a range of angle.
    1. Get angles of all points.
        * If there is no reference point to compute angle, use centroid of points by default.
        * https://github.com/hihi313/Computer-Vision-and-Applications/blob/3d98242967ab2efe023a6a9305a532f790561221/Midterm/feet_measure.py#L266-L277
    2. Select points whithin range: angle +- epsilon.
        * Epsilon is (candidate) region to find extreme point of foot
        * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L286-L288
    3. Select extreme point (based on y axis value) from candidate region
        * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L299-L300


## Get half of foot from both views

1. Get line function that cut through the foot.
    * By the obtained extreme points from previous step.
    * Line = top $\times$ bottom.
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L339-L344
2. Find boundary points (intersection of line & image boundary).
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L360-L364
3. Get mask by filling regions of left or right boundary(the line).
    * Fill the polygon consists of boundary points.
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L365-L369
4. Get half of foot from the mask.
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L377-L378

## Combine

It can be viewed as using the 2 half-non-foot region to cut out the shape of a (whole) foot.
Then invert the image & get the contour of the foot.

1. Shift(align) image.
    * Shift the right part of foot here, the shifting function:
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L385-L390
2. Combine.
    * Add the (cut-out) regions, then inverse.
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L393-L394
3. Get foot contour for following operations
    * Get the largesst contour
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L399-L402

# Measure the foot

1. Get the (rotated) bounding rectangle of the (foot) contour
    * Return values are: (x, y), (width, height), rotation angle
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L408
2. Draw bounding box & foot
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L413-L420
3. Convert the bounding box's height & width to `cm`.
    * By ratio: `<width or height>px of bounding box / <width or height>px of A4 sheet * <width or height>cm/inch of A4 sheet`.
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L423-L430
4. Draw the measured values
    * https://github.com/hihi313/Computer-Vision-and-Applications/blob/492189e8c00d3b325f17075cddf7a4ee1f83ccda/Midterm/feet_measure.py#L435-L440