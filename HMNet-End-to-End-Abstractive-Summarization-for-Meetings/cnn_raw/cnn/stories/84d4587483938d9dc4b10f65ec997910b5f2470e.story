(Mashable) -- This is what the world looks like, according to the Facebook social graph.

Facebook intern Paul Butler was interested in the locations of friendships, so he decided to create a visualization of Facebook connections around the globe. How local are our friends? Where are the highest concentration of friendships? How do political and geological boundaries affect them?

Butler started by using a sample of 10 million friend pairs, correlated them with their current cities and then mapped that data using the longitude and latitude of each city. That was the easy part. Creating the right effect to show connecting relationships between thousands of cities proved to be a challenge. Butler wrote a fascinating Facebook note explaining some of the challenges he faced creating his visualization:

"I began exploring it in R, an open-source statistics environment. As a sanity check, I plotted points at some of the latitude and longitude coordinates. To my relief, what I saw was roughly an outline of the world. Next I erased the dots and plotted lines between the points.

After a few minutes of rendering, a big white blob appeared in the center of the map. Some of the outer edges of the blob vaguely resembled the continents, but it was clear that I had too much data to get interesting results just by drawing lines. I thought that making the lines semi-transparent would do the trick, but I quickly realized that my graphing environment couldn't handle enough shades of color for it to work the way I wanted.

Instead I found a way to simulate the effect I wanted. I defined weights for each pair of cities as a function of the Euclidean distance between them and the number of friends between them. Then I plotted lines between the pairs by weight, so that pairs of cities with the most friendships between them were drawn on top of the others.

I used a color ramp from black to blue to white, with each line's color depending on its weight. I also transformed some of the lines to wrap around the image, rather than spanning more than halfway around the world."

With a few more tweaks, he eventually came up with the amazing visualization you see here. At first glance, it provides some expected data -- the U.S. has the highest concentration of Facebook friendships, and Africa has the lowest concentration. While most of Russia and Antarctica are nowhere to be found, the rest of the world is easily identifiable.

What do you think of the visualization? Does anything about it surprise you? Let us know in the comments.

© 2010 MASHABLE.com. All rights reserved.

@highlight

A Facebook intern created a visualization of Facebook connections around the globe

@highlight

Using a sample of 10 million friend pairs, he correlated them with their current cities

@highlight

The U.S. has the highest concentration of Facebook friendships while Africa has the lowest