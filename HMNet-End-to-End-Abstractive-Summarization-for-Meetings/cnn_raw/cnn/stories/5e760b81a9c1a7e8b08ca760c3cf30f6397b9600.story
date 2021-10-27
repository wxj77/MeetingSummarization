(Ars Technica) -- Wikipedia's "crowdsourced knowledge" model has created a spectacular resource, but everyone knows the big caveat: If the data's important, don't trust the online encyclopedia without verifying it first.

So how well would a similar crowdsourcing model work for a detailed street-level map of the world?

Several years ago, that question led to the creation of OpenStreetMap.org, a wiki-style world map that anyone can edit. Several friends in the UK got fed up with the government's stingy approach to data distribution (Ordnance Survey has crafted ultra-detailed maps of the UK with public funds, but that data must be licensed for private use), and decided to do something about it.

The obvious question was "why bother?" Terrific maps from Google, Microsoft, and others already had a significant lead and were often free (as in beer) to embed and use. But location services were exploding, and all were premised on mapping data. The lack of any high-quality, worldwide, free, and open-source map remained a problem.

"Most hackers around the world are familiar with the difference between 'free' as in 'free beer' and as in 'free speech,'" says OpenStreetMap in its list of frequently asked questions.

"Google Maps are free as in beer, not as in speech. If your project's mapping needs can be served simply by using the Google Maps API, all to the good. But that's not true of every project. We need a free dataset which will enable programmers, social activists, cartographers and the like to fulfill their plans without being limited either by Google's API or by their Terms of Service."

A street-level map of the world might sound audacious, but OpenStreetMap has exploded in popularity-after starting with a few friends, more than 250,000 have now contributed mapping data to the project. Soon, the map achieved amazing accuracy, especially in Europe where it originated.

Take a look at Germany, for instance, where open mapping has become a craze. The Zoologischer Garten Berlin (the Berlin Zoo) exists in Google Maps, of course, but it has little detail (though it does have satellite maps, which OpenStreetMap lacks).

Dedicated locals have used OpenStreetMap's tools to do Google one better by mapping all of the zoo's animals; if you want to plot your visit to the lair of the "Gro_er Panda," you can. Even restroom locations are helpfully plotted.

As the map took off, it became clear that something more was needed to truly enable developers. The map data alone was valuable, but how much more valuable would it be to create a mapping platform that could handle commercial loads, provide back-end routing services, handle geocoding and reverse geocoding, and build tools for working with the data and crafting new apps from it?

Thus CloudMade was born. After a year of building out the platform (with much of the work being done by Ukrainian developers), CloudMade now has 10,500 developers using its mapping platform. Every week, the platform pulls the newest data from OpenStreetMap, creating something new: the ability for frustrated map users to correct those annoying local errors, then see the changes propagate out to apps within a week.

Corrections are made "by people who know their area," says CloudMade VP Christian Petersen. And while one might assume that the bulk of the work being done is in places like the US or Europe, Petersen says that "67 percent of all the mapping being done right now is outside of those markets."

CloudMade hopes to profit by offering free access to its platform in exchange for a cut of ad revenue from free applications (devs can also pay upfront if they want).

Where available, baseline map data was imported from open datasets, like the US government's TIGER data from the Census Bureau. But in many places like Europe, most of the map was handcrafted and began with a blank canvas. The results are impressive: a spin around the map shows fairly detailed work already done on places like Mumbai and La Paz, though truly remote locations like South Georgia island have no features yet.

Unexpected challenges popped up along the way. In China, for instance, the state puts tough limits on private mapping. "Doing business in China is challenging" says Petersen.

And the occasional edit fray breaks out over sensitive maps, such as those depicting the divided island of Cyprus.

But Petersen believes that the crowdsourced approach to the underlying data works well-better than commercial alternatives, in fact. "Passion wins," he says, noting that pro mapping companies might send data collectors into an area once a year or so and update their maps even less frequently. When local users are involved, local changes are made quickly.

Clean up your neighborhood

The accuracy of the data was put to the test last week when a company called skobbler released a free turn-by-turn, GPS-based map program for the iPhone based on the CloudMade platform. Given the price of competing GPS navigation software, this sounds like a revolution.

Sadly, the program doesn't work well. Crashes were common in our testing, response could be slow, and the interface was non-intuitive. Users have given it a two-star rating. Even the official press release featured a telling quote: "Although we know we're not quite ready to challenge the expensive premium navigation solutions, we'll quickly get there," said skobbler co-founder Marcus Thielking.

Such problems can be fixed. But there's a deeper issue: will consumers trust a mapping program that encourages them to tap a "ladybug button" every time they run into a map problem? (The button sets a flag in OpenStreetMap that locals can use to investigate and correct errors.)

Users may balk at helping to create the map they're relying on, but people once said the same thing about Wikipedia. Certainly, the underlying maps are getting a workout; CloudMade says that 7,017 edits are made every hour.

The process is addicting. A quick look at my own neighborhood revealed a tiny error-a road mistakenly extended up a private driveway about a block away. I created an account at OpenStreetMap, zoomed into the problem area, and hit "edit." A Flash-based map editor popped up, overlaying OpenStreetMap data on a satellite picture of the area pulled from Yahoo. Fixing the area was a simple matter of dragging and clicking, and voila-I had made a small difference in the world.

Twenty minutes later, after cleaning up pond boundaries at a nearby park, adding a fire station, and fixing a road that mistakenly drove through some homes, I reluctantly moved on to other things. The map's level of detail is already superb, and editing was enjoyable.

Having such a free and open resource on the Web seems like a Good Thing. Now, if CloudMade can partner with some truly outstanding developers and turn out some well-written code, it could soon be an extremely Useful Thing, too.

COPYRIGHT 2010 ARSTECHNICA.COM

@highlight

OpenStreetMap aims to let users create map data free for anyone

@highlight

More than 250,00 users have added or edited street-level maps of the world

@highlight

The map's iPhone app has some issues, however