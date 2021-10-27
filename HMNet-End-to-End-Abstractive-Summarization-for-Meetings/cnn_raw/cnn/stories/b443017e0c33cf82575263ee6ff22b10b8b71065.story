(WIRED) -- You should care about Apple's collection of geodata on iPhones, iPads and iPod Touch devices, because the method is flawed.

To be clear, "care" doesn't mean you should smash your iPhone with a hammer, rip out the GPS chip and gulp it down your throat. This isn't an issue of "Big Brother is watching."

It's just a matter of a security flaw that puts your location data at risk if it gets in the wrong hands -- not an immediate concern, but a concern nonetheless.

Two data scientists broke the news Wednesday that an unencrypted file stored on iOS devices contains a detailed log of the device's geographical data dating back 10 months. The scientists also wrote a program, allowing you to plug in your iOS device and automatically output the geodata into an interactive map, just so you could see for yourself.

As this story developed, some tech observers have attempted to defuse the issue. "So what?" David Pogue wrote in his New York Times column. "I have nothing to hide. Who cares if anyone knows where I've been?"

Here's why we care.

Permanent data storage is unnecessary

As WIRED.com pointed out yesterday, Apple already admitted and explained that it deliberately stores geodata on its mobile devices so the company can collect it to improve location services.

The general process, summarized: Whenever you use an app with a location service -- the Yelp app, for instance, to find nearby restaurants -- the iPhone gets information about nearby cell towers and Wi-Fi access points, and stores the info.

Every 12 hours, an iOS device's stored geodata gets anonymized with a random string of numbers, and it gets transmitted to Apple in a batch. Apple says it keeps all this data in its own database, so it can provide you quicker and more precise location services.

So when you use a location-based app, such as your Maps app to get your location, you're first pulling data from Apple's geo database to get your general location, and then your GPS chip homes in on a more precise latitude and longitude. Apple's location database speeds up the location process.

Location gathering techniques like this aren't anything new. For instance, when using an app like Google Maps app on your Android phone, some of your location data is cached -- or stored -- so that if your network connection is interrupted, following directions on the map won't be.

Data caching also improves the speed of an app's performance.

"It makes such a huge difference when you can cache this data," Andreas Schobel, CTO of Android-app--developing studio Catch.com. "Cellphone connections are incredibly high in latency. Imagine having to wait half-a-second longer when sending a tweet with your location included. From a user experience point of view, these caches make sense."

But the problem remains that there is no reason for that geo data to remain on your device after it's transmitted to Apple.

In contrast to Apple, Google's stance on the position is clear: It has been upfront about location data collection from the start. In a statement provided to WIRED.com, Google says as much:

All location sharing on Android is opt-in by the user. We provide users with notice and control over the collection, sharing and use of location in order to provide a better mobile experience on Android devices. Any location data that is sent back to Google location servers is anonymized and is not tied or traceable to a specific user.

This is true for both Android and iPhones, but it's no longer the point. Having a data file with over a year's worth of your location information stored on your iPhone is a security risk.

So if a thief got his hands on your iPhone, he can figure out where you live and loot you there, too. Same goes for a hacker who gains remote access to the consolidated.db file.

But if a thief or hacker dug into an Android device, there isn't going to be much geodata saved on the smartphone to digitally stalk you. (There's plenty of other data on smartphones such as text messages, address books and so forth, but at least we have control over what data we store in this regard.)

Bottom line, this data shouldn't stick around on your iOS device, because it does nothing but put you at risk. And you should care about that, because this problem can be and should be fixed by Apple, and you should demand that.

The database makes a tempting target for law enforcement

If police wanted to, they could subpoena the iPhone's location database file when investigating a suspect. That file contains too much information for this to even be justified.

Imagine if you were suspected of a crime and police wanted to know where you were at 5 p.m. Thursday. They could subpoena your iPhone, dig into this file and, looking at the various data points, get a good idea of where you were at that time.

Sure, that sounds like it could be a useful practice for busting bad criminals, but what about all that other data? With that file police can not only find out where you were at 5 p.m. Thursday, but also that you see a therapist every Monday morning, or simply that you were somewhere that you'd want to keep to yourself -- private matters.

As tempting as it may be to say, "They're suspected for a crime, they deserve it," even suspects deserve privacy. They're suspects, after all, not criminals (yet). The fact that law enforcement can easily get more information than necessary is not a positive thing.

But it's not a huge immediate danger

With that said, the chances are small that your iPhone is going to get hacked or stolen, or that you're going to be suspected of a crime (we would hope).

So there's no reason to freak out. But we should care about the implications of a rich file of geographic data living on our iOS devices offering no customer benefit, creating digital footprints that we can't erase.

Fortunately, Apple is a media giant, and customer trust is too valuable for the company to lose. It's likely we'll see Apple issue a software update soon tweaking the geodata-storage method, hopefully with a full explanation.

Subscribe to WIRED magazine for less than $1 an issue and get a FREE GIFT! Click here!

Copyright 2011 Wired.com.

@highlight

If a thief got his hands on your iPhone, he can figure out where you live

@highlight

The police could subpoena the iPhone's location database file when investigating a suspect

@highlight

It's likely we'll see Apple issue a software update soon